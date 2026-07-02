import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
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

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'conversation_log', label: 'Conversation Log' },
  { id: 'daily_summary', label: 'Daily Summary' },
  { id: 'topics', label: 'Topic Analysis' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ConversationalAIDashboard() {
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
      axios.get(`${API_URL}/api/conversational-ai/overview`),
      axios.get(`${API_URL}/api/conversational-ai/breakdown`),
      axios.get(`${API_URL}/api/conversational-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Conversational AI data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No conversational AI data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Conversational AI Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Clinical conversation analysis, turn tracking, topic extraction, and dialogue quality monitoring
      </p>

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

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'conversation_log' && <ConversationLogTab breakdown={breakdown} />}
      {tab === 'daily_summary' && <DailySummaryTab breakdown={breakdown} />}
      {tab === 'topics' && <TopicAnalysisTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = overview.kpis || []
  const roleDist = overview.role_distribution || []
  const turnLengthDist = overview.turn_length_distribution || []
  const dailyActivity = overview.daily_activity || []
  const hourlyActivity = overview.response_time_stats || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color || COLORS[i % COLORS.length]} /></Card>
      ))}

      {/* Role Distribution Pie */}
      <Card title="Role Distribution" span={2}>
        {roleDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={roleDist} dataKey="count" nameKey="label" cx="50%" cy="50%" outerRadius={100}
                label={({ label, count }) => `${label} (${count})`}>
                {roleDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No role distribution data</div>}
      </Card>

      {/* Turn Length Distribution Bar */}
      <Card title="Turn Length Distribution" span={2}>
        {turnLengthDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={turnLengthDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {turnLengthDist.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No turn length data</div>}
      </Card>

      {/* Daily Activity Line Chart */}
      <Card title="Daily Conversation Activity" span={2}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="operator_turns" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Operator" />
              <Line type="monotone" dataKey="assistant_turns" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Assistant" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily activity data</div>}
      </Card>

      {/* Hourly Activity Bar Chart */}
      <Card title="Hourly Activity" span={2}>
        {hourlyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={hourlyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No hourly activity data</div>}
      </Card>
    </div>
  )
}

function ConversationLogTab({ breakdown }) {
  const conversations = breakdown?.conversation_log || []
  const [search, setSearch] = useState('')

  const filtered = search
    ? conversations.filter(c =>
        (c.text || '').toLowerCase().includes(search.toLowerCase()) ||
        (c.role || '').toLowerCase().includes(search.toLowerCase())
      )
    : conversations

  const ROLE_COLORS = { operator: '#3b82f6', assistant: '#10b981' }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Conversation Log (${filtered.length})`}>
        <div style={{ marginBottom: 12 }}>
          <input
            type="text"
            placeholder="Search conversations..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0',
              fontSize: 13, outline: 'none', boxSizing: 'border-box'
            }}
          />
        </div>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['ID', 'Role', 'Text', 'Timestamp', 'Words', 'Chars'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Words' || h === 'Chars' ? 'right' : 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{c.id}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={c.role} color={ROLE_COLORS[c.role] || '#6b7280'} />
                  </td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {c.text || ''}
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{c.ts_local || ''}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{c.word_count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{c.char_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DailySummaryTab({ breakdown }) {
  const daily = breakdown?.daily_summary || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Daily Summary (${daily.length} days)`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Date', 'Operator Turns', 'Assistant Turns', 'Total Turns', 'Total Words'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Date' ? 'left' : 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {daily.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.date}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#3b82f6' }}>{d.operator_turns}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#10b981' }}>{d.assistant_turns}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{d.total_turns}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{d.total_words}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function TopicAnalysisTab({ breakdown }) {
  const topics = (breakdown?.topics || []).slice(0, 20)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Top Topic Keywords (${topics.length})`}>
        {topics.length > 0 ? (
          <ResponsiveContainer width="100%" height={Math.max(400, topics.length * 32)}>
            <BarChart data={topics} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="keyword" type="category" tick={{ fontSize: 11 }} width={160} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                {topics.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No topic data available</div>}
      </Card>
    </div>
  )
}

function PipelineTab({ breakdown }) {
  const events = breakdown?.pipeline_events || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Conversational AI Pipeline Events (${events.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Action', 'Detail', 'Timestamp'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{e.action}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || ''}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{e.ts || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div></Card>

  const concepts = definitions.concepts || []
  const qualityMetrics = definitions.quality_metrics || []
  const interactionTypes = definitions.interaction_types || []
  const compliance = definitions.compliance || []
  const remediation = definitions.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Conversational AI Concepts */}
      <Card title={`Conversational AI Concepts (${concepts.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Term</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Interaction Types */}
      <Card title={`Interaction Types (${interactionTypes.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Type</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {interactionTypes.map((t, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{t.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{t.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Quality Metrics */}
      <Card title={`Quality Metrics (${qualityMetrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {qualityMetrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Compliance */}
      <Card title={`Compliance References (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Standard</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Notes</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.ref}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Remediation */}
      <Card title={`Remediation Strategies (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Strategy</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{r.strategy}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
