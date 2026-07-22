import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', planned: '#f97316' }

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const bg = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

function FlowPipeline({ steps }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 4 }}>
      {steps.map((step, i) => (
        <React.Fragment key={i}>
          <span style={{
            background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe',
            borderRadius: 6, padding: '4px 10px', fontSize: 11, fontWeight: 500
          }}>{step}</span>
          {i < steps.length - 1 && <span style={{ color: '#94a3b8', fontSize: 14 }}>&rarr;</span>}
        </React.Fragment>
      ))}
    </div>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function TabScaffoldDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/tab-scaffold/overview`),
      axios.get(`${API_URL}/api/tab-scaffold/breakdown`),
      axios.get(`${API_URL}/api/tab-scaffold/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Tab Scaffold...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>Tab scaffold data not available</div>

  const tabs = ['overview', 'tabs', 'default-template', 'flows', 'definitions']
  const k = overview.kpis || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
          <KPI label="Tabs" value={k.total_tabs} />
          <KPI label="Sections per Tab" value={k.total_sections} />
          <KPI label="Built" value={k.built} />
          <KPI label="Planned" value={k.planned} />
          <KPI label="Total ToDos" value={k.total_todos} />
          <KPI label="Total Flow Steps" value={k.total_flow_steps} />
          <KPI label="Default Sections" value={k.default_sections} />
        </div>
      </Card>

      <Card title="Tab Status Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
              {(overview.status_distribution || []).map((_, i) => (
                <Cell key={i} fill={STATUS_COLORS[overview.status_distribution[i]?.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Flow Steps per Tab">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={overview.flow_per_tab} margin={{ left: 10, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Flow Steps" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="ToDos per Tab">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={overview.todos_per_tab} margin={{ left: 10, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#22c55e" name="ToDos" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="All Tabs Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Tab</th>
                <th style={thStyle}>Goal</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>ToDos</th>
                <th style={thStyle}>Flow Steps</th>
              </tr>
            </thead>
            <tbody>
              {(overview.tab_summary || []).map((t, i) => (
                <tr key={i}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{t.id}</td>
                  <td style={{ ...tdStyle, maxWidth: 300 }}>{t.goal}</td>
                  <td style={tdStyle}><StatusBadge status={t.status} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{t.todos}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{t.flow_steps}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  const renderTabs = () => {
    if (!breakdown) return null
    const tabEntries = Object.entries(breakdown.tabs || {})
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
        {tabEntries.map(([tid, t]) => (
          <Card key={tid} title={
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {tid.replace(/_/g, ' ')} <StatusBadge status={t.status || 'planned'} />
            </span>
          }>
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 8 }}><strong>Goal:</strong> {t.goal}</div>
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 4 }}>Process Flow</div>
                <FlowPipeline steps={t.flow || []} />
              </div>
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 4 }}>ToDos</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {(t.todos || []).map((todo, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 2 }}>{todo}</li>
                  ))}
                </ul>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 2 }}>Input</div>
                  <div style={{ fontSize: 12, color: '#334155' }}>{t.input}</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 2 }}>Process</div>
                  <div style={{ fontSize: 12, color: '#334155' }}>{t.process}</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 2 }}>Output</div>
                  <div style={{ fontSize: 12, color: '#334155' }}>{t.output}</div>
                </div>
              </div>
              {t.viz && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 2 }}>Visualization</div>
                  <div style={{ fontSize: 12, color: '#334155' }}>{t.viz}</div>
                </div>
              )}
            </div>
          </Card>
        ))}
      </div>
    )
  }

  const renderDefaultTemplate = () => {
    if (!breakdown) return null
    const d = breakdown.default || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
        <Card title="Default 8-Section Template" span={2}>
          <p style={{ fontSize: 12, color: '#475569', marginTop: 0 }}>
            {breakdown.note || 'This default template is applied to every tab unless a per-tab override exists.'}
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#3b82f6', marginBottom: 4 }}>Goal</div>
              <div style={{ fontSize: 12, color: '#334155' }}>{d.goal}</div>
            </div>
            <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#3b82f6', marginBottom: 4 }}>Input</div>
              <div style={{ fontSize: 12, color: '#334155' }}>{d.input}</div>
            </div>
            <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#3b82f6', marginBottom: 4 }}>Process</div>
              <div style={{ fontSize: 12, color: '#334155' }}>{d.process}</div>
            </div>
            <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#3b82f6', marginBottom: 4 }}>Output</div>
              <div style={{ fontSize: 12, color: '#334155' }}>{d.output}</div>
            </div>
            <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#3b82f6', marginBottom: 4 }}>Visualization</div>
              <div style={{ fontSize: 12, color: '#334155' }}>{d.viz}</div>
            </div>
            <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#3b82f6', marginBottom: 4 }}>ToDos</div>
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {(d.todos || []).map((todo, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 2 }}>{todo}</li>
                ))}
              </ul>
            </div>
          </div>
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#3b82f6', marginBottom: 6 }}>Default Flow</div>
            <FlowPipeline steps={d.flow || []} />
          </div>
        </Card>
      </div>
    )
  }

  const renderFlows = () => {
    if (!breakdown) return null
    const tabEntries = Object.entries(breakdown.tabs || {})
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Default Flow">
          <FlowPipeline steps={(breakdown.default || {}).flow || []} />
        </Card>
        {tabEntries.map(([tid, t]) => (
          <Card key={tid} title={
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {tid.replace(/_/g, ' ')} <StatusBadge status={t.status || 'planned'} />
            </span>
          }>
            <FlowPipeline steps={t.flow || []} />
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>{t.goal}</div>
          </Card>
        ))}
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!defs) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
        <Card title="Status Legend">
          {(defs.status_legend || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <StatusBadge status={s.status} />
              <span style={{ fontSize: 12, color: '#475569' }}>{s.meaning}</span>
            </div>
          ))}
        </Card>

        <Card title="Glossary" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
            {(defs.glossary || []).map((g, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 6, padding: '8px 12px' }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#1e293b' }}>{g.term}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{g.definition}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Clinical Notes">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {(defs.notes || []).map((n, i) => (
              <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{n}</li>
            ))}
          </ul>
        </Card>

        <Card title="References">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {(defs.references || []).map((r, i) => (
              <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4, fontFamily: 'monospace' }}>{r}</li>
            ))}
          </ul>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Tab Scaffold Dashboard</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        Standard 8-section scaffold pattern — {k.total_tabs} tabs, {k.total_sections} sections each, {k.total_flow_steps} total flow steps
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
            border: tab === t ? '2px solid #3b82f6' : '1px solid #e2e8f0',
            background: tab === t ? '#eff6ff' : '#fff',
            color: tab === t ? '#1d4ed8' : '#64748b'
          }}>
            {t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
          </button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'tabs' && renderTabs()}
      {tab === 'default-template' && renderDefaultTemplate()}
      {tab === 'flows' && renderFlows()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
