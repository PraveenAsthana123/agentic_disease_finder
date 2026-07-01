import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from 'recharts'

const API = '/api'

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

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const CAT_COLORS = { clinical: '#ef4444', technical: '#3b82f6', operational: '#10b981' }

export default function PromptDriftDashboard() {
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
      axios.get(`${API}/prompt-drift/overview`),
      axios.get(`${API}/prompt-drift/breakdown`),
      axios.get(`${API}/prompt-drift/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'volume', label: 'Volume & Distribution' },
    { id: 'topics', label: 'Topic Analysis' },
    { id: 'weekly', label: 'Weekly Drift' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Prompt Drift dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No prompt drift data available.'}</div>

  const driftColor = (v) => {
    const abs = Math.abs(parseFloat(v) || 0)
    return abs > 50 ? '#ef4444' : abs > 20 ? '#f59e0b' : '#10b981'
  }

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Prompt Drift Monitor</h2>
        <Badge text={`${overview.kpis?.[0]?.value || 0} prompts tracked`} color="#3b82f6" />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>Last run: {overview.run_at}</span>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'volume' && renderVolume()}
      {tab === 'topics' && renderTopics()}
      {tab === 'weekly' && renderWeekly()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpis = overview.kpis || []
    const lengthData = overview.length_over_time || []
    const roleDist = overview.role_distribution || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((k, i) => (
          <Card key={i}>
            <KPI
              label={k.label}
              value={typeof k.value === 'number' ? (k.unit === 'chars' ? k.value.toLocaleString() : k.value) : k.value}
              sub={k.unit}
              color={k.label.includes('Drift') ? driftColor(k.value) : undefined}
            />
          </Card>
        ))}

        <Card title="Prompt Drift Interpretation" span={4}>
          <p style={{ margin: 0, fontSize: 14, color: '#475569', lineHeight: 1.6 }}>{overview.interpretation}</p>
        </Card>

        <Card title="Prompt & Response Length Over Time" span={3}>
          {lengthData.length > 0 && (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={lengthData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(0) + ' chars' : v} />
                <Legend />
                <Line type="monotone" dataKey="avg_prompt_len" stroke="#3b82f6" strokeWidth={2} name="Avg Prompt Len" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="avg_response_len" stroke="#10b981" strokeWidth={2} name="Avg Response Len" dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Role Distribution" span={1}>
          {roleDist.length > 0 && (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={roleDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {roleDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderVolume() {
    if (!breakdown?.available) return null
    const promptHist = breakdown.prompt_length_histogram || []
    const responseHist = breakdown.response_length_histogram || []
    const dailyVol = breakdown.daily_volume || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Prompt Length Distribution">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={promptHist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Prompts" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Response Length Distribution">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={responseHist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} name="Responses" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Daily Message Volume" span={2}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={dailyVol}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="prompts" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Prompts" />
              <Bar dataKey="responses" fill="#10b981" radius={[4, 4, 0, 0]} name="Responses" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Prompt File Library" span={2}>
          <div style={{ maxHeight: 400, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Filename</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>Length (chars)</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.prompt_file_stats || []).map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 12 }}>{f.filename}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{f.length.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderTopics() {
    if (!breakdown?.available) return null
    const keywords = breakdown.topic_keywords || []
    const clinical = keywords.filter(k => k.category === 'clinical')
    const technical = keywords.filter(k => k.category === 'technical')
    const operational = keywords.filter(k => k.category === 'operational')

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Top 20 Keywords from Operator Prompts">
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={keywords} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="keyword" width={90} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} name="Frequency">
                {keywords.map((k, i) => <Cell key={i} fill={CAT_COLORS[k.category] || '#94a3b8'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card title={`Clinical (${clinical.length})`}>
            {clinical.map((k, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ fontSize: 13, color: '#1e293b' }}>{k.keyword}</span>
                <Badge text={k.count} color="#ef4444" />
              </div>
            ))}
            {clinical.length === 0 && <p style={{ color: '#94a3b8', fontSize: 13 }}>None in top 20</p>}
          </Card>

          <Card title={`Technical (${technical.length})`}>
            {technical.map((k, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ fontSize: 13, color: '#1e293b' }}>{k.keyword}</span>
                <Badge text={k.count} color="#3b82f6" />
              </div>
            ))}
            {technical.length === 0 && <p style={{ color: '#94a3b8', fontSize: 13 }}>None in top 20</p>}
          </Card>

          <Card title={`Operational (${operational.length})`}>
            {operational.map((k, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                <span style={{ fontSize: 13, color: '#1e293b' }}>{k.keyword}</span>
                <Badge text={k.count} color="#10b981" />
              </div>
            ))}
            {operational.length === 0 && <p style={{ color: '#94a3b8', fontSize: 13 }}>None in top 20</p>}
          </Card>
        </div>

        <Card title="Topic Category Distribution">
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={[
                  { name: 'Clinical', value: clinical.reduce((s, k) => s + k.count, 0) },
                  { name: 'Technical', value: technical.reduce((s, k) => s + k.count, 0) },
                  { name: 'Operational', value: operational.reduce((s, k) => s + k.count, 0) },
                ].filter(d => d.value > 0)}
                dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                <Cell fill="#ef4444" />
                <Cell fill="#3b82f6" />
                <Cell fill="#10b981" />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>
    )
  }

  function renderWeekly() {
    if (!breakdown?.available) return null
    const weekly = breakdown.weekly_drift || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Weekly Prompt & Response Length Drift">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={weekly} margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(0) : v} />
              <Legend />
              <Line type="monotone" dataKey="avg_prompt_len" stroke="#3b82f6" strokeWidth={2} name="Avg Prompt Len" dot={{ r: 4 }} />
              <Line type="monotone" dataKey="avg_response_len" stroke="#10b981" strokeWidth={2} name="Avg Response Len" dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Weekly Volume">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={weekly}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="n_prompts" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Prompts" />
              <Bar dataKey="n_responses" fill="#10b981" radius={[4, 4, 0, 0]} name="Responses" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Weekly Detail Table">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px' }}>Week</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Prompts</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Responses</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Avg Prompt Len</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Avg Response Len</th>
              </tr>
            </thead>
            <tbody>
              {weekly.map((w, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontFamily: 'monospace' }}>{w.week}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{w.n_prompts}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{w.n_responses}</td>
                  <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{w.avg_prompt_len?.toFixed(0)}</td>
                  <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{w.avg_response_len?.toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions?.sections) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {definitions.sections.map((sec, i) => (
          <Card key={i} title={sec.title}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <tbody>
                {sec.items.map((item, j) => (
                  <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600, width: '25%', verticalAlign: 'top', color: '#334155' }}>{item.term}</td>
                    <td style={{ padding: '8px 6px', color: '#475569', lineHeight: 1.5 }}>{item.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        ))}
      </div>
    )
  }
}
