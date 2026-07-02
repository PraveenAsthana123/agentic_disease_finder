import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function RoleBadge({ role }) {
  const colors = {
    operator: { bg: '#dbeafe', fg: '#1e40af', border: '#bfdbfe' },
    assistant: { bg: '#dcfce7', fg: '#166534', border: '#bbf7d0' }
  }
  const c = colors[role] || { bg: '#f1f5f9', fg: '#475569', border: '#e2e8f0' }
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, color: c.fg, background: c.bg,
      padding: '2px 8px', borderRadius: 10, border: `1px solid ${c.border}`
    }}>{(role || 'unknown').toUpperCase()}</span>
  )
}

export default function GenerativeAIDashboard() {
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
          axios.get(`${API_URL}/generative-ai/overview`),
          axios.get(`${API_URL}/generative-ai/breakdown`),
          axios.get(`${API_URL}/generative-ai/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load generative AI data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading generative AI analytics...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'conversations', label: 'Conversations' },
    { id: 'activity', label: 'Activity Patterns' },
    { id: 'genai-queries', label: 'GenAI Queries' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Generative AI Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        AI conversation analytics, GenAI bot usage, content safety, and responsible AI governance
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} breakdown={breakdown} />}
      {tab === 'conversations' && <ConversationsTab breakdown={breakdown} />}
      {tab === 'activity' && <ActivityTab overview={overview} breakdown={breakdown} />}
      {tab === 'genai-queries' && <GenAIQueriesTab overview={overview} breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview, breakdown }) {
  const safetyColor = overview.content_safety_score >= 0.95 ? '#10b981' :
    overview.content_safety_score >= 0.85 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total Messages">
        <KPI label="Conversation Log" value={overview.total_messages} color="#3b82f6" />
      </Card>
      <Card title="Assistant Responses">
        <KPI label="AI-Generated" value={overview.assistant_messages} sub={`Ratio: ${overview.response_ratio}x`} color="#10b981" />
      </Card>
      <Card title="Operator Queries">
        <KPI label="Human Input" value={overview.operator_messages} color="#8b5cf6" />
      </Card>
      <Card title="GenAI Bot Queries">
        <KPI label="Ollama-Powered" value={overview.genai_bot_queries} color="#f59e0b" />
      </Card>
      <Card title="Content Safety Score">
        <KPI label="Safety Rating" value={(overview.content_safety_score * 100).toFixed(1) + '%'} sub={`${overview.flagged_messages} flagged`} color={safetyColor} />
      </Card>
      <Card title="Avg Response Length">
        <KPI label="Assistant (chars)" value={overview.avg_assistant_response_length} color="#06b6d4" />
      </Card>
      <Card title="Avg Query Length">
        <KPI label="Operator (chars)" value={overview.avg_operator_query_length} color="#ec4899" />
      </Card>
      <Card title="AI Components Active">
        <KPI label="Logged in Transactions" value={(overview.ai_component_usage || []).length} color="#64748b" />
      </Card>

      <Card title="Daily Conversation Volume" span={2}>
        {(overview.daily_trend || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={overview.daily_trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Area type="monotone" dataKey="messages" stroke="#3b82f6" fill="#3b82f680" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily trend data</div>}
      </Card>

      <Card title="AI Component Usage" span={2}>
        {(overview.ai_component_usage || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={overview.ai_component_usage} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis dataKey="component" type="category" tick={{ fontSize: 10 }} width={100} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No AI component data</div>}
      </Card>
    </div>
  )
}

function ConversationsTab({ breakdown }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Message Length Distribution">
        {(breakdown.length_distribution || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={breakdown.length_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No length data</div>}
      </Card>

      <Card title="Recent Conversations (Last 20)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>ID</th>
                <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Role</th>
                <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Preview</th>
                <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {(breakdown.recent_conversations || []).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px' }}>{c.id}</td>
                  <td style={{ padding: '6px' }}><RoleBadge role={c.role} /></td>
                  <td style={{ padding: '6px', maxWidth: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.preview}</td>
                  <td style={{ padding: '6px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{c.ts_local}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function ActivityTab({ overview, breakdown }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Hourly Activity Pattern" span={2}>
        {(breakdown.hourly_pattern || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={breakdown.hourly_pattern}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="messages" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No hourly data</div>}
      </Card>

      <Card title="Daily Messages by Role" span={2}>
        {(breakdown.daily_by_role || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={breakdown.daily_by_role}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="operator" fill="#8b5cf6" stackId="a" radius={[0, 0, 0, 0]} />
              <Bar dataKey="assistant" fill="#10b981" stackId="a" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily role data</div>}
      </Card>

      <Card title="AI Transaction Breakdown" span={2}>
        {(breakdown.ai_transactions || []).length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Component</th>
                  <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Action</th>
                  <th style={{ padding: '8px 6px', textAlign: 'right', color: '#64748b' }}>Count</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.ai_transactions || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 500 }}>{t.component}</td>
                    <td style={{ padding: '6px' }}>{t.action}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontWeight: 600 }}>{t.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No transaction data</div>}
      </Card>
    </div>
  )
}

function GenAIQueriesTab({ overview, breakdown }) {
  const roleData = Object.entries(overview.role_usage || {}).map(([k, v]) => ({ role: k, count: v }))
  const layoutData = Object.entries(overview.layout_usage || {}).map(([k, v]) => ({ layout: k, count: v }))
  const contentTypes = breakdown.content_type_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="GenAI Bot Role Usage">
        {roleData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={roleData} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={70} label={({ role, count }) => `${role}: ${count}`}>
                {roleData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 40 }}>No role usage data yet</div>}
      </Card>

      <Card title="Output Layout Distribution">
        {layoutData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={layoutData} dataKey="count" nameKey="layout" cx="50%" cy="50%" outerRadius={70} label={({ layout, count }) => `${layout}: ${count}`}>
                {layoutData.map((_, i) => <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 40 }}>No layout data yet</div>}
      </Card>

      <Card title="GenAI Query Log" span={2}>
        {(breakdown.genai_queries || []).length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Patient</th>
                  <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Query Detail</th>
                  <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b' }}>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.genai_queries || []).map((q, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 500 }}>{q.patient_id}</td>
                    <td style={{ padding: '6px' }}>{q.detail}</td>
                    <td style={{ padding: '6px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{q.ts_local}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No GenAI bot queries recorded yet. Queries will appear here when the Ollama-powered bot is used.</div>}
      </Card>

      <Card title="Content Type Distribution" span={2}>
        {contentTypes.length > 0 ? (
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={contentTypes}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="type" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>Content type data will populate with GenAI bot usage</div>}
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs || !defs.sections) return <div style={{ color: '#94a3b8' }}>No definitions available</div>
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {defs.sections.map((sec, si) => (
        <Card key={si} title={sec.title}>
          <div style={{ display: 'grid', gap: 10 }}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ borderBottom: ii < sec.items.length - 1 ? '1px solid #f1f5f9' : 'none', paddingBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
