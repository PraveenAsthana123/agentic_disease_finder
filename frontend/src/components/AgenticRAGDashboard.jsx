import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TIER_COLORS = { full: '#10b981', partial: '#f59e0b', sparse: '#ef4444' }
const HEALTH_COLORS = { healthy: '#10b981', degraded: '#f59e0b', empty: '#ef4444', critical: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'corpus', label: 'Corpus & Coverage' },
  { id: 'agents', label: 'Agent Traces' },
  { id: 'health', label: 'KB Health' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AgenticRAGDashboard() {
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
      axios.get(`${API}/api/agentic-rag/overview`),
      axios.get(`${API}/api/agentic-rag/breakdown`),
      axios.get(`${API}/api/agentic-rag/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Agentic RAG data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const k = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Agentic RAG Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Retrieval-Augmented Generation pipeline analytics from real clinical knowledge base
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab k={k} overview={overview} />}
      {tab === 'corpus' && <CorpusTab overview={overview} breakdown={breakdown} />}
      {tab === 'agents' && <AgentTab breakdown={breakdown} overview={overview} />}
      {tab === 'health' && <HealthTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ k, overview }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI Row */}
      <Card><KPI label="Total Documents" value={k.total_documents?.toLocaleString()} sub="across all sources" color="#3b82f6" /></Card>
      <Card><KPI label="Total Chunks" value={k.total_chunks?.toLocaleString()} sub="indexed for retrieval" color="#8b5cf6" /></Card>
      <Card><KPI label="Active Agents" value={k.active_agents} sub={`${k.query_types} query types`} color="#10b981" /></Card>
      <Card><KPI label="Avg Latency" value={`${k.avg_retrieval_latency_ms}ms`} sub="retrieval pipeline" color="#f59e0b" /></Card>
      <Card><KPI label="Source Tables" value={k.source_tables} sub="clinical data sources" color="#06b6d4" /></Card>
      <Card><KPI label="Full Coverage" value={`${k.coverage_full_pct}%`} sub="patients with 4+ sources" color="#10b981" /></Card>
      <Card><KPI label="KB Health" value={k.kb_health} sub="knowledge base status" color={HEALTH_COLORS[k.kb_health] || '#64748b'} /></Card>
      <Card><KPI label="Query Types" value={k.query_types} sub="routed to agents" color="#ec4899" /></Card>

      {/* Coverage Distribution Pie */}
      <Card title="Retrieval Coverage Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.coverage_distribution || []} dataKey="count" nameKey="tier" cx="50%" cy="50%" outerRadius={80} label={({ tier, count }) => `${tier}: ${count}`}>
              {(overview.coverage_distribution || []).map((d, i) => (
                <Cell key={i} fill={TIER_COLORS[d.tier] || COLORS[i]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Query Routing Bar */}
      <Card title="Query Routing — Retrievable Docs by Type" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.query_routing || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="label" width={130} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="retrievable_docs" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Corpus Inventory Table */}
      <Card title="Corpus Inventory" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Source</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Description</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Documents</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Chunks</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Avg Chunks/Doc</th>
              </tr>
            </thead>
            <tbody>
              {(overview.corpus_inventory || []).map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{row.source_table}</td>
                  <td style={{ padding: 8, color: '#64748b' }}>{row.description}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{row.document_count}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{row.chunk_count}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{row.avg_chunks_per_doc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function CorpusTab({ overview, breakdown }) {
  const coverage = breakdown?.patient_coverage || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Coverage by patient */}
      <Card title="Per-Patient Coverage (%)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={coverage.slice(0, 40)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={60} />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Bar dataKey="coverage_pct" fill="#3b82f6" radius={[4, 4, 0, 0]}>
              {coverage.slice(0, 40).map((d, i) => (
                <Cell key={i} fill={TIER_COLORS[d.tier] || '#3b82f6'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Patient coverage table */}
      <Card title="Patient Coverage Detail" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Sources Present</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Coverage %</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Tier</th>
              </tr>
            </thead>
            <tbody>
              {coverage.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{p.sources_present} / {p.total_sources}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{p.coverage_pct}%</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>
                    <Badge text={p.tier} color={TIER_COLORS[p.tier] || '#64748b'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Query routing detail */}
      <Card title="Query Routing — Latency by Type" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={overview?.query_routing || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={60} />
            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="avg_latency_ms" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function AgentTab({ breakdown, overview }) {
  const traces = breakdown?.agent_traces || []
  const workload = breakdown?.agent_workload || []
  const relevance = breakdown?.relevance_by_query_type || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Agent workload pie */}
      <Card title="Agent Workload Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={workload} dataKey="trace_count" nameKey="agent" cx="50%" cy="50%" outerRadius={80} label={({ agent, trace_count }) => `${agent.split('_').join(' ')}: ${trace_count}`}>
              {workload.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Relevance by query type */}
      <Card title="Avg Relevance Score by Query Type">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={relevance}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="query_type" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="avg_relevance" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Agent pipeline traces table */}
      <Card title="Recent Agent Pipeline Traces" span={2}>
        <div style={{ maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Trace ID</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Agent</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Query Type</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Chunks</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Relevance</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Total ms</th>
              </tr>
            </thead>
            <tbody>
              {traces.map((t, i) => {
                const totalMs = (t.steps || []).reduce((s, st) => s + st.duration_ms, 0)
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{t.trace_id}</td>
                    <td style={{ padding: 8 }}><Badge text={t.agent} color={COLORS[i % COLORS.length]} /></td>
                    <td style={{ padding: 8, color: '#64748b' }}>{t.query_type}</td>
                    <td style={{ padding: 8, textAlign: 'right' }}>{t.chunks_retrieved}</td>
                    <td style={{ padding: 8, textAlign: 'right', color: t.relevance_score >= 0.8 ? '#10b981' : '#f59e0b' }}>
                      {t.relevance_score}
                    </td>
                    <td style={{ padding: 8, textAlign: 'right' }}>{totalMs}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Query routing agents */}
      <Card title="Agent Routing Map" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {(overview?.query_routing || []).map((q, i) => (
            <div key={i} style={{
              padding: 12, borderRadius: 8, background: '#f8fafc',
              border: '1px solid #e2e8f0'
            }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{q.label}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>
                Agent: <Badge text={q.agent} color={COLORS[i % COLORS.length]} />
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                Sources: {q.source_tables.join(', ')}
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                {q.retrievable_docs} docs | {q.avg_latency_ms}ms
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function HealthTab({ breakdown }) {
  const checks = breakdown?.knowledge_base_health || []
  const summary = breakdown?.health_summary || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      <Card><KPI label="Healthy Sources" value={summary.healthy} color="#10b981" /></Card>
      <Card><KPI label="Degraded Sources" value={summary.degraded} color="#f59e0b" /></Card>
      <Card><KPI label="Empty Sources" value={summary.empty} color="#ef4444" /></Card>

      {/* Health detail table */}
      <Card title="Knowledge Base Health Detail" span={3}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Source</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Description</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Records</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Empty Fields %</th>
              <th style={{ textAlign: 'center', padding: 8 }}>Health</th>
            </tr>
          </thead>
          <tbody>
            {checks.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{c.source}</td>
                <td style={{ padding: 8, color: '#64748b' }}>{c.description}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{c.record_count}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{c.empty_field_pct}%</td>
                <td style={{ padding: 8, textAlign: 'center' }}>
                  <Badge text={c.health} color={HEALTH_COLORS[c.health] || '#64748b'} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Empty field % bar chart */}
      <Card title="Empty Field % by Source" span={3}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={checks.filter(c => c.record_count > 0)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="source" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={60} />
            <YAxis domain={[0, 100]} label={{ value: '%', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="empty_field_pct" radius={[4, 4, 0, 0]}>
              {checks.filter(c => c.record_count > 0).map((c, i) => (
                <Cell key={i} fill={c.empty_field_pct > 30 ? '#ef4444' : c.empty_field_pct > 15 ? '#f59e0b' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {(definitions.sections || []).map((sec, si) => (
        <Card key={si} title={sec.title}>
          <div style={{ display: 'grid', gap: 12 }}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
