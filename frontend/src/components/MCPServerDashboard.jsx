import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const fmt = v => (v != null ? v : '--')
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'tools', label: 'Tools' },
  { id: 'resources', label: 'Resources' },
  { id: 'prompts', label: 'Prompts' },
  { id: 'definitions', label: 'Definitions' },
]

export default function MCPServerDashboard() {
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
      axios.get(`${API_URL}/api/mcp-server/overview`),
      axios.get(`${API_URL}/api/mcp-server/breakdown`),
      axios.get(`${API_URL}/api/mcp-server/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading MCP Server Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>MCP Server Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Model Context Protocol server — exposing clinical AI tools, resources, and prompts to external agents
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#334155', fontWeight: tab === t.id ? 600 : 400,
            fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── OVERVIEW TAB ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card>
            <KPI label="Server Status" value={fmt(ov?.server_status)} sub="Current state" color={ov?.server_status === 'running' ? '#10b981' : '#ef4444'} />
          </Card>
          <Card>
            <KPI label="Total Tools" value={fmt(ov?.total_tools)} sub="Registered tools" color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Total Resources" value={fmt(ov?.total_resources)} sub="Exposed resources" color="#8b5cf6" />
          </Card>
          <Card>
            <KPI label="Total Prompts" value={fmt(ov?.total_prompts)} sub="Prompt templates" color="#f59e0b" />
          </Card>
          <Card>
            <KPI label="Connected Clients" value={fmt(ov?.connected_clients)} sub="Active connections" color="#06b6d4" />
          </Card>
          <Card>
            <KPI label="Requests Served" value={fmt(ov?.requests_served)} sub="Total handled" color="#10b981" />
          </Card>

          {/* Tool Categories Pie Chart */}
          <Card title="Tool Categories" span={1}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={(ov?.tool_categories || []).filter(d => d.value > 0)}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.tool_categories || []).filter(d => d.value > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Resource Types Bar Chart */}
          <Card title="Resource Types" span={1}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={ov?.resource_types || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" name="Resources" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {(ov?.resource_types || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Capability Radar Chart */}
          <Card title="Capability Radar" span={1}>
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={ov?.capability_radar || []}>
                <PolarGrid />
                <PolarAngleAxis dataKey="capability" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar name="Score" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          {/* Server Info Card */}
          <Card title="Server Info" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, fontSize: 13 }}>
              <div>
                <div style={{ color: '#64748b', marginBottom: 4 }}>Server Name</div>
                <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(ov?.server_name)}</div>
              </div>
              <div>
                <div style={{ color: '#64748b', marginBottom: 4 }}>Protocol Version</div>
                <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(ov?.protocol_version)}</div>
              </div>
              <div>
                <div style={{ color: '#64748b', marginBottom: 4 }}>Uptime</div>
                <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(ov?.uptime)}</div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* ─── TOOLS TAB ─── */}
      {tab === 'tools' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Tools Table */}
          <Card title="Registered Tools" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Name</th>
                    <th style={{ padding: '8px 12px' }}>Category</th>
                    <th style={{ padding: '8px 12px' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.tools || []).map((tool, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{fmt(tool.name)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: '#ede9fe', color: '#6d28d9'
                        }}>{fmt(tool.category)}</span>
                      </td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{fmt(tool.description)}</td>
                    </tr>
                  ))}
                  {(bd?.tools || []).length === 0 && (
                    <tr><td colSpan={3} style={{ padding: '12px', color: '#94a3b8', textAlign: 'center' }}>No tools registered</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Tools Per Category Bar Chart */}
          <Card title="Tools Per Category" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={
                Object.entries(
                  (bd?.tools || []).reduce((acc, t) => {
                    const cat = t.category || 'Uncategorized'
                    acc[cat] = (acc[cat] || 0) + 1
                    return acc
                  }, {})
                ).map(([category, count]) => ({ category, count }))
              }>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" name="Tools" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                  {Object.keys(
                    (bd?.tools || []).reduce((acc, t) => {
                      acc[t.category || 'Uncategorized'] = true
                      return acc
                    }, {})
                  ).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── RESOURCES TAB ─── */}
      {tab === 'resources' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Resources Table */}
          <Card title="Exposed Resources" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Name</th>
                    <th style={{ padding: '8px 12px' }}>URI</th>
                    <th style={{ padding: '8px 12px' }}>Type</th>
                    <th style={{ padding: '8px 12px' }}>MIME Type</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.resources || []).map((res, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{fmt(res.name)}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12, color: '#475569' }}>{fmt(res.uri)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: '#dbeafe', color: '#1e40af'
                        }}>{fmt(res.type)}</span>
                      </td>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12, color: '#64748b' }}>{fmt(res.mime_type)}</td>
                    </tr>
                  ))}
                  {(bd?.resources || []).length === 0 && (
                    <tr><td colSpan={4} style={{ padding: '12px', color: '#94a3b8', textAlign: 'center' }}>No resources exposed</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Resources By Type Pie Chart */}
          <Card title="Resources By Type" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={
                    Object.entries(
                      (bd?.resources || []).reduce((acc, r) => {
                        const t = r.type || 'Unknown'
                        acc[t] = (acc[t] || 0) + 1
                        return acc
                      }, {})
                    ).map(([name, value]) => ({ name, value }))
                  }
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {Object.keys(
                    (bd?.resources || []).reduce((acc, r) => {
                      acc[r.type || 'Unknown'] = true
                      return acc
                    }, {})
                  ).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── PROMPTS TAB ─── */}
      {tab === 'prompts' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Prompts Table */}
          <Card title="Prompt Templates" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Name</th>
                    <th style={{ padding: '8px 12px' }}>Description</th>
                    <th style={{ padding: '8px 12px' }}>Arguments</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.prompts || []).map((prompt, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{fmt(prompt.name)}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{fmt(prompt.description)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        {Array.isArray(prompt.arguments) ? prompt.arguments.map((arg, j) => (
                          <span key={j} style={{
                            display: 'inline-block', marginRight: 4, marginBottom: 2,
                            padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                            background: '#fef3c7', color: '#92400e'
                          }}>{arg}</span>
                        )) : <span style={{ color: '#94a3b8', fontSize: 12 }}>{fmt(prompt.arguments)}</span>}
                      </td>
                    </tr>
                  ))}
                  {(bd?.prompts || []).length === 0 && (
                    <tr><td colSpan={3} style={{ padding: '12px', color: '#94a3b8', textAlign: 'center' }}>No prompts registered</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Transport Info Card */}
          <Card title="Transport Configuration" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, fontSize: 13 }}>
              <div>
                <div style={{ color: '#64748b', marginBottom: 4 }}>Transport Type</div>
                <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(bd?.transport?.type)}</div>
              </div>
              <div>
                <div style={{ color: '#64748b', marginBottom: 4 }}>Host</div>
                <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(bd?.transport?.host)}</div>
              </div>
              <div>
                <div style={{ color: '#64748b', marginBottom: 4 }}>Port</div>
                <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(bd?.transport?.port)}</div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {(defs?.terms || []).map((d, i) => (
            <Card key={i} title={d.term}>
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{d.definition}</p>
              {d.category && (
                <span style={{
                  display: 'inline-block', marginTop: 8, padding: '2px 8px', borderRadius: 4,
                  fontSize: 11, background: '#ede9fe', color: '#6d28d9'
                }}>{d.category}</span>
              )}
            </Card>
          ))}
          {(defs?.terms || []).length === 0 && (
            <Card span={2}><p style={{ color: '#94a3b8' }}>No definitions available</p></Card>
          )}
        </div>
      )}
    </div>
  )
}
