import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function AgentMemoryDashboard() {
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
          axios.get(`${API_URL}/agent-memory/overview`),
          axios.get(`${API_URL}/agent-memory/breakdown`),
          axios.get(`${API_URL}/agent-memory/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load agent memory data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8, animation: 'spin 1.5s linear infinite' }}>&#9881;</div>
      Loading agent memory data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Agent memory data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const domainFill = overview.domain_fill_rates || []
  const compDist = overview.completeness_distribution || []
  const dailyTrend = overview.daily_activity_trend || []
  const writePatterns = overview.memory_write_patterns || []
  const staleness = overview.staleness || []
  const convCtx = overview.conversation_context || {}

  const profiles = breakdown?.patient_profiles || []
  const cooccurrence = breakdown?.domain_cooccurrence || []
  const gaps = breakdown?.coverage_gaps || []
  const compAttr = breakdown?.component_attribution || []
  const actorAttr = breakdown?.actor_attribution || []
  const diseaseDepth = breakdown?.disease_memory_depth || []
  const recentWrites = breakdown?.recent_memory_writes || []
  const definitions = defs?.definitions || []

  const cardStyle = { background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', marginBottom: 18 }
  const kpiStyle = { background: '#f8fafc', borderRadius: 10, padding: '14px 18px', minWidth: 140, textAlign: 'center' }
  const sectionTitle = { fontSize: 15, fontWeight: 700, color: '#1e293b', marginBottom: 12 }
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: '8px 8px 0 0', fontWeight: active ? 700 : 400,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#475569',
    border: 'none', fontSize: 13, marginRight: 4
  })
  const thStyle = { padding: '8px 12px', textAlign: 'left', fontSize: 12, color: '#64748b', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '8px 12px', fontSize: 13, color: '#334155', borderBottom: '1px solid #f1f5f9' }

  const kpiItems = [
    { label: 'Memory Coverage', value: `${fmt(s.memory_coverage_pct)}%` },
    { label: 'Avg Completeness', value: `${fmt(s.avg_completeness_pct)}%` },
    { label: 'Total Records', value: fmt(s.total_memory_records) },
    { label: 'Patients', value: fmt(s.total_patients) },
    { label: 'With Memory', value: fmt(s.patients_with_memory) },
    { label: 'Conversations', value: fmt(s.total_conversations) },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patient Memory' },
    { id: 'gaps', label: 'Gaps & Attribution' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Agent Memory Dashboard</h2>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>real clinical.db per-patient memory analytics</span>
      </div>

      {/* Tab bar */}
      <div style={{ marginBottom: 18 }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══ OVERVIEW TAB ═══ */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {kpiItems.map((k, i) => (
              <div key={i} style={kpiStyle}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Charts row: Domain Fill Rate Bar + Completeness Pie */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Domain Fill Rates (%)</h4>
              {domainFill.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={domainFill} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} fontSize={11} />
                    <YAxis dataKey="domain" type="category" fontSize={11} width={110} />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Bar dataKey="coverage_pct" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                      {domainFill.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No domain data</div>
              )}
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Completeness Distribution</h4>
              {compDist.length > 0 ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ResponsiveContainer width="55%" height={240}>
                    <PieChart>
                      <Pie data={compDist} dataKey="count" nameKey="bucket" cx="50%" cy="50%" innerRadius={50} outerRadius={90}
                        label={({ bucket, count }) => `${bucket}: ${count}`}>
                        {compDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                  <div style={{ marginLeft: 16 }}>
                    {compDist.map((q, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <div style={{ width: 12, height: 12, borderRadius: 3, background: COLORS[i % COLORS.length] }} />
                        <span style={{ fontSize: 13, color: '#334155' }}>{q.bucket}: {fmt(q.count)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No completeness data</div>
              )}
            </div>
          </div>

          {/* Daily Activity Trend */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Daily Memory Activity (Transactions)</h4>
            {dailyTrend.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={dailyTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Line type="monotone" dataKey="transactions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No daily activity data</div>
            )}
          </div>

          {/* Conversation Context + Write Patterns */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Conversation Context</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <tbody>
                  {[
                    ['Total Turns', convCtx.total_turns],
                    ['Assistant', convCtx.assistant],
                    ['User', convCtx.user],
                    ['System', convCtx.system],
                    ['Avg Response Length', `${fmt(s.avg_response_length)} chars`],
                  ].map(([label, val], i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontWeight: 600, width: '50%' }}>{label}</td>
                      <td style={tdStyle}>{fmt(val)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Memory Write Patterns (Top Actions)</h4>
              {writePatterns.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={writePatterns.slice(0, 8)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="action" fontSize={10} angle={-20} textAnchor="end" height={50} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No write pattern data</div>
              )}
            </div>
          </div>

          {/* Staleness table */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Memory Staleness (Latest Update per Domain)</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Domain</th>
                  <th style={thStyle}>Latest Update</th>
                </tr>
              </thead>
              <tbody>
                {staleness.map((st, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{st.domain}</td>
                    <td style={tdStyle}>{st.latest_update || <span style={{ color: '#94a3b8' }}>never</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ═══ PATIENT MEMORY TAB ═══ */}
      {tab === 'patients' && (
        <>
          {/* Per-patient profiles */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Per-Patient Memory Completeness ({profiles.length} patients)</h4>
            <div style={{ maxHeight: 480, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Dept</th>
                    <th style={thStyle}>Domains</th>
                    <th style={thStyle}>Completeness</th>
                    <th style={thStyle}>Filled Domains</th>
                  </tr>
                </thead>
                <tbody>
                  {profiles.map((p, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{p.name || p.patient_id}</td>
                      <td style={tdStyle}>{p.disease || '--'}</td>
                      <td style={tdStyle}>{p.department || '--'}</td>
                      <td style={tdStyle}>{p.domains_filled}/{p.total_domains}</td>
                      <td style={tdStyle}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <div style={{ width: 60, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              width: `${p.completeness_pct}%`, height: '100%', borderRadius: 4,
                              background: p.completeness_pct >= 50 ? '#10b981' : p.completeness_pct >= 25 ? '#f59e0b' : '#ef4444'
                            }} />
                          </div>
                          <span style={{ fontSize: 12 }}>{p.completeness_pct}%</span>
                        </div>
                      </td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{p.filled_domains.join(', ') || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Disease Memory Depth */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Memory Depth by Disease</h4>
            {diseaseDepth.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={diseaseDepth}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" fontSize={11} angle={-15} textAnchor="end" height={50} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="total_records" fill="#8b5cf6" name="Total Records" radius={[4, 4, 0, 0]}>
                    {diseaseDepth.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No disease data</div>
            )}
          </div>

          {/* Domain co-occurrence */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Domain Co-occurrence (patients with both domains)</h4>
            {cooccurrence.length > 0 ? (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Domain A</th>
                      <th style={thStyle}>Domain B</th>
                      <th style={thStyle}>Patients</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cooccurrence.slice(0, 20).map((c, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{c.domain_a}</td>
                        <td style={tdStyle}>{c.domain_b}</td>
                        <td style={tdStyle}>{c.patients_with_both}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 40 }}>No co-occurrence data</div>
            )}
          </div>
        </>
      )}

      {/* ═══ GAPS & ATTRIBUTION TAB ═══ */}
      {tab === 'gaps' && (
        <>
          {/* Coverage gaps */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Coverage Gaps (lowest fill-rate domains)</h4>
            {gaps.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Domain</th>
                    <th style={thStyle}>Patients Missing</th>
                    <th style={thStyle}>Fill Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {gaps.map((g, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{g.domain}</td>
                      <td style={tdStyle}>{g.patients_missing}</td>
                      <td style={tdStyle}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <div style={{ width: 80, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              width: `${g.fill_rate_pct}%`, height: '100%', borderRadius: 4,
                              background: g.fill_rate_pct >= 50 ? '#10b981' : g.fill_rate_pct >= 20 ? '#f59e0b' : '#ef4444'
                            }} />
                          </div>
                          <span style={{ fontSize: 12 }}>{g.fill_rate_pct}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 40 }}>All domains fully covered</div>
            )}
          </div>

          {/* Component + Actor attribution side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Component Attribution (memory writes)</h4>
              {compAttr.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={compAttr.slice(0, 10)} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" fontSize={11} />
                    <YAxis dataKey="component" type="category" fontSize={11} width={110} />
                    <Tooltip />
                    <Bar dataKey="writes" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                      {compAttr.slice(0, 10).map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No attribution data</div>
              )}
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Actor Attribution (who writes memory)</h4>
              {actorAttr.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <Pie data={actorAttr} dataKey="writes" nameKey="actor" cx="50%" cy="50%" innerRadius={50} outerRadius={90}
                      label={({ actor, writes }) => `${actor}: ${writes}`}>
                      {actorAttr.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No actor data</div>
              )}
            </div>
          </div>

          {/* Recent memory writes */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Recent Memory Writes (last 20 transactions)</h4>
            {recentWrites.length > 0 ? (
              <div style={{ maxHeight: 360, overflow: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>ID</th>
                      <th style={thStyle}>Patient</th>
                      <th style={thStyle}>Component</th>
                      <th style={thStyle}>Action</th>
                      <th style={thStyle}>Actor</th>
                      <th style={thStyle}>Detail</th>
                      <th style={thStyle}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentWrites.map((w, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{w.id}</td>
                        <td style={tdStyle}>{w.patient_id || '--'}</td>
                        <td style={tdStyle}>{w.component || '--'}</td>
                        <td style={tdStyle}>{w.action || '--'}</td>
                        <td style={tdStyle}>{w.actor || '--'}</td>
                        <td style={{ ...tdStyle, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{w.detail || '--'}</td>
                        <td style={{ ...tdStyle, fontSize: 11 }}>{w.ts || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 40 }}>No recent writes</div>
            )}
          </div>
        </>
      )}

      {/* ═══ DEFINITIONS TAB ═══ */}
      {tab === 'definitions' && (
        <div style={cardStyle}>
          <h4 style={sectionTitle}>Metric Definitions</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ ...thStyle, width: '25%' }}>Metric</th>
                <th style={thStyle}>Definition</th>
              </tr>
            </thead>
            <tbody>
              {definitions.map((d, i) => (
                <tr key={i}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{d.metric}</td>
                  <td style={tdStyle}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
