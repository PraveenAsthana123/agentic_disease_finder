import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STATUS_COLORS = { built: '#10b981', planned: '#3b82f6', partial: '#f59e0b', unknown: '#94a3b8' }
const HEALTH_COLORS = { healthy: '#10b981', degraded: '#f59e0b', unhealthy: '#ef4444', unknown: '#94a3b8' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function MCPGovernanceDashboard() {
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
          axios.get(`${API_URL}/api/mcp-governance/overview`),
          axios.get(`${API_URL}/api/mcp-governance/breakdown`),
          axios.get(`${API_URL}/api/mcp-governance/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load MCP governance data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading MCP governance data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'MCP governance data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const agentStatusDist = overview.agent_status_distribution || []
  const moduleGroups = overview.module_groups || []
  const pipeStatus = overview.pipeline_status || []
  const daily = overview.daily_activity || []
  const actionDist = overview.action_distribution || []
  const agents = breakdown?.agent_inventory || []
  const pipeGroups = breakdown?.pipeline_breakdown || []
  const consensus = breakdown?.consensus_records || []
  const definitions = defs?.definitions || []
  const complianceRefs = defs?.compliance_refs || []
  const remediation = defs?.remediation || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#475569',
    border: active ? '1px solid #2563eb' : '1px solid #cbd5e1'
  })
  const healthDot = (status) => ({
    display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
    background: HEALTH_COLORS[status] || '#94a3b8', marginRight: 6
  })
  const statusBadge = (st) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${STATUS_COLORS[st] || '#94a3b8'}22`, color: STATUS_COLORS[st] || '#64748b'
  })

  const kpiItems = [
    { label: 'Total Agents', value: s.total_agents, color: '#3b82f6' },
    { label: 'Certified (Built)', value: s.built_agents, color: '#10b981' },
    { label: 'Certification %', value: s.certified_pct != null ? `${s.certified_pct}%` : '--', color: '#10b981' },
    { label: 'Pipelines Built', value: `${s.built_pipelines}/${s.total_pipelines}`, color: '#8b5cf6' },
    { label: 'Pipeline Gov %', value: s.pipeline_governance_pct != null ? `${s.pipeline_governance_pct}%` : '--', color: '#8b5cf6' },
    { label: 'MCP Messages', value: s.n_messages, color: '#06b6d4' },
    { label: 'Audit Entries', value: s.n_audit_entries, color: '#f59e0b' },
    { label: 'Blocked Events', value: s.blocked_events, color: '#ef4444' },
  ]

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 22, color: '#1e293b' }}>MCP Governance Dashboard</h2>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 18, flexWrap: 'wrap' }}>
        {[
          ['overview', 'Overview'],
          ['agents', 'Agent Inventory'],
          ['pipelines', 'Pipeline Governance'],
          ['protocol', 'MCP Protocol'],
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

          {/* Health status */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>System Health</h4>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              <div>
                <span style={healthDot(s.mcp_health)}></span>
                <strong>MCP Protocol:</strong> {s.mcp_health}
              </div>
              <div>
                <span style={healthDot(s.a2a_health)}></span>
                <strong>Agent-to-Agent:</strong> {s.a2a_health}
              </div>
              <div><strong>Pattern:</strong> {s.mcp_pattern}</div>
              <div><strong>Disease Agents:</strong> {s.n_disease_agents}</div>
            </div>
            {overview.mcp_recommendations?.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#f59e0b', marginBottom: 4 }}>Recommendations:</div>
                {overview.mcp_recommendations.map((r, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 2 }}>- {r}</div>
                ))}
              </div>
            )}
          </div>

          {/* Agent status + Module groups charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Agent Status Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={agentStatusDist} dataKey="count" nameKey="status" cx="50%" cy="50%"
                    innerRadius={50} outerRadius={85}
                    label={({ status, count }) => `${status}: ${count}`}>
                    {agentStatusDist.map((d, i) => (
                      <Cell key={i} fill={STATUS_COLORS[d.status] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Top Module Groups</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={moduleGroups.slice(0, 8)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="group" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Agents" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Daily activity + Action distribution */}
          {daily.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Daily Transaction Activity</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={daily}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Events" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {actionDist.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Action Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={actionDist}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="action" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {tab === 'agents' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Agent Inventory ({agents.length} agents)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>ID</th>
                  <th style={{ padding: '8px 10px' }}>Task</th>
                  <th style={{ padding: '8px 10px' }}>Module</th>
                  <th style={{ padding: '8px 10px' }}>Group</th>
                  <th style={{ padding: '8px 10px' }}>Status</th>
                  <th style={{ padding: '8px 10px' }}>Certified</th>
                </tr>
              </thead>
              <tbody>
                {agents.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{a.id}</td>
                    <td style={{ padding: '8px 10px', color: '#1e293b', maxWidth: 300 }}>{a.task}</td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.module}</td>
                    <td style={{ padding: '8px 10px' }}>{a.group}</td>
                    <td style={{ padding: '8px 10px' }}><span style={statusBadge(a.status)}>{a.status}</span></td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{a.certified ? '\u2705' : '\u2B1C'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'pipelines' && (
        <>
          {/* Pipeline status chart */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Pipeline Status Overview</h4>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pipeStatus} dataKey="count" nameKey="status" cx="50%" cy="50%"
                  innerRadius={50} outerRadius={85}
                  label={({ status, count }) => `${status}: ${count}`}>
                  {pipeStatus.map((d, i) => (
                    <Cell key={i} fill={STATUS_COLORS[d.status] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {pipeGroups.map((g, gi) => (
            <div key={gi} style={cardStyle}>
              <h4 style={{ margin: '0 0 8px', color: '#334155' }}>
                {g.group} <span style={{ fontSize: 13, color: '#64748b', fontWeight: 400 }}>
                  ({g.built}/{g.total} built)
                </span>
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 10 }}>
                {g.pipelines.map((p, pi) => (
                  <div key={pi} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <strong style={{ color: '#1e293b', fontSize: 13 }}>{p.name}</strong>
                      <span style={statusBadge(p.status)}>{p.status}</span>
                    </div>
                    {p.maps_to && <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{p.maps_to}</div>}
                    <div style={{ fontSize: 11, color: '#94a3b8' }}>
                      Stages: {p.stages.join(' \u2192 ')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </>
      )}

      {tab === 'protocol' && (
        <>
          {/* MCP + A2A health */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>MCP Protocol Health</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
              <div style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 16 }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                  <span style={healthDot(s.mcp_health)}></span> MCP Status: {s.mcp_health}
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Messages: {s.n_messages}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Audit Entries: {s.n_audit_entries}</div>
                {overview.mcp_issues?.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: '#ef4444' }}>Issues:</div>
                    {overview.mcp_issues.map((issue, i) => (
                      <div key={i} style={{ fontSize: 11, color: '#ef4444' }}>- {issue}</div>
                    ))}
                  </div>
                )}
              </div>
              <div style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 16 }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                  <span style={healthDot(s.a2a_health)}></span> A2A Status: {s.a2a_health}
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Pattern: {s.mcp_pattern}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Disease Agents: {s.n_disease_agents}</div>
                {overview.a2a_issues?.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: '#f59e0b' }}>Issues:</div>
                    {overview.a2a_issues.map((issue, i) => (
                      <div key={i} style={{ fontSize: 11, color: '#f59e0b' }}>- {issue}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Consensus voting */}
          {consensus.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Agent Consensus Voting</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Disease Agent</th>
                    <th style={{ padding: '8px 10px' }}>Classification</th>
                    <th style={{ padding: '8px 10px' }}>Probability</th>
                  </tr>
                </thead>
                <tbody>
                  {consensus.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.disease}</td>
                      <td style={{ padding: '8px 10px' }}>{c.label}</td>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{(c.probability * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {tab === 'definitions' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>MCP Governance Concepts</h4>
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
        Source: agent_tasks.json (60 agents), agentic_mcp_report.json, enterprise_pipelines.json, clinical.db transaction_log
      </div>
    </div>
  )
}
