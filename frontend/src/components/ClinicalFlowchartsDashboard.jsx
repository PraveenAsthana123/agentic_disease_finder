import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const COMPLEXITY_COLORS = { linear: '#10b981', medium: '#f59e0b', high: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
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

function KPI({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
    </div>
  )
}

function ComplexityBadge({ complexity }) {
  const c = (complexity || '').toLowerCase()
  const bg = c === 'linear' ? '#dcfce7' : c === 'medium' ? '#fef3c7' : c === 'high' ? '#fee2e2' : '#e2e8f0'
  const fg = c === 'linear' ? '#166534' : c === 'medium' ? '#92400e' : c === 'high' ? '#991b1b' : '#475569'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: bg, color: fg
    }}>{complexity || 'unknown'}</span>
  )
}

function TypeBadge({ type }) {
  const t = (type || '').toLowerCase()
  let bg = '#e2e8f0', fg = '#475569'
  if (t === 'decision') { bg = '#fef3c7'; fg = '#92400e' }
  if (t === 'action' || t === 'process') { bg = '#dbeafe'; fg = '#1e40af' }
  if (t === 'start') { bg = '#dcfce7'; fg = '#166534' }
  if (t === 'end') { bg = '#fee2e2'; fg = '#991b1b' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: bg, color: fg
    }}>{type || 'unknown'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'detail', label: 'Detail' },
  { id: 'mermaid', label: 'Mermaid' },
]

export default function ClinicalFlowchartsDashboard() {
  const [overview, setOverview] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [selectedId, setSelectedId] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/clinical-flowcharts/overview`),
      axios.get(`${API_URL}/api/clinical-flowcharts/analytics`),
    ])
      .then(([oRes, aRes]) => {
        setOverview(oRes.data)
        setAnalytics(aRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  const loadDetail = async (id) => {
    try {
      const res = await axios.get(`${API_URL}/api/clinical-flowcharts/detail/${id}`)
      setDetail(res.data)
      setSelectedId(id)
      setTab('detail')
    } catch (e) {
      setError(e.message)
    }
  }

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Clinical Flowcharts...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const flowcharts = overview?.flowcharts || []
  const an = overview?.analytics || {}

  // Analytics tab data
  const complexityData = analytics?.complexity_distribution
    ? Object.entries(analytics.complexity_distribution).map(([name, value]) => ({
        name, value, fill: COMPLEXITY_COLORS[name] || '#64748b'
      }))
    : []
  const perFlowchart = analytics?.per_flowchart || []
  const categoryDist = analytics?.category_distribution || []
  const processTypes = analytics?.process_types || []

  const renderOverview = () => (
    <>
      {/* Summary KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
        <Card>
          <KPI label="Total Flowcharts" value={overview?.total} color="#3b82f6" />
        </Card>
        <Card>
          <KPI label="Total Nodes" value={an.total_nodes} color="#10b981" />
        </Card>
        <Card>
          <KPI label="Decision Points" value={an.total_decisions} color="#f59e0b" />
        </Card>
        <Card>
          <KPI label="Total Edges" value={an.total_edges} color="#8b5cf6" />
        </Card>
      </div>

      {/* Flowchart cards */}
      <Card title="Flowcharts">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 14 }}>
          {flowcharts.map((f, i) => (
            <div
              key={f.id || i}
              onClick={() => loadDetail(f.id)}
              style={{
                border: '1px solid #e2e8f0', borderRadius: 10, padding: 16, cursor: 'pointer',
                transition: 'box-shadow 0.15s', background: '#fafbfc'
              }}
              onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,.1)'}
              onMouseLeave={e => e.currentTarget.style.boxShadow = 'none'}
            >
              <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 6 }}>{f.title}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>{f.category}</div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <ComplexityBadge complexity={f.complexity} />
                <span style={{ fontSize: 11, color: '#64748b' }}>Nodes: {fmt(f.node_count)}</span>
                <span style={{ fontSize: 11, color: '#64748b' }}>Decisions: {fmt(f.decision_count)}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </>
  )

  const renderAnalytics = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      {/* Complexity Distribution */}
      <Card title="Complexity Distribution">
        {complexityData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={complexityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                {complexityData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>

      {/* Nodes per Flowchart */}
      <Card title="Nodes per Flowchart">
        {perFlowchart.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perFlowchart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="title" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="node_count" fill="#3b82f6" name="Nodes" />
              <Bar dataKey="decision_count" fill="#f59e0b" name="Decisions" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>

      {/* Category Distribution */}
      <Card title="Category Distribution">
        {categoryDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={categoryDist} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="category" type="category" tick={{ fontSize: 11 }} width={120} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6">
                {categoryDist.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>

      {/* Process Types */}
      <Card title="Process Types">
        {processTypes.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={processTypes} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={90} label={({ type, count }) => `${type}: ${count}`}>
                {processTypes.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>
    </div>
  )

  const renderDetail = () => {
    if (!detail) return (
      <Card>
        <div style={{ textAlign: 'center', color: '#64748b', padding: 40 }}>
          Select a flowchart from the Overview tab or click a button below.
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center', marginTop: 16 }}>
            {flowcharts.map((f, i) => (
              <button key={f.id || i} onClick={() => loadDetail(f.id)} style={{
                padding: '6px 14px', borderRadius: 6, border: '1px solid #e2e8f0',
                background: '#f8fafc', cursor: 'pointer', fontSize: 12, color: '#334155'
              }}>{f.title}</button>
            ))}
          </div>
        </div>
      </Card>
    )

    const nodes = detail.nodes || []

    return (
      <>
        {/* Selector buttons */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
          {flowcharts.map((f, i) => (
            <button key={f.id || i} onClick={() => loadDetail(f.id)} style={{
              padding: '6px 14px', borderRadius: 6, border: selectedId === f.id ? '2px solid #3b82f6' : '1px solid #e2e8f0',
              background: selectedId === f.id ? '#eff6ff' : '#f8fafc', cursor: 'pointer', fontSize: 12,
              fontWeight: selectedId === f.id ? 700 : 400, color: '#334155'
            }}>{f.title}</button>
          ))}
        </div>

        {/* Detail header */}
        <Card title={detail.title}>
          <div style={{ display: 'flex', gap: 24, marginBottom: 16 }}>
            <KPI label="Nodes" value={detail.node_count} color="#3b82f6" />
            <KPI label="Decisions" value={detail.decision_count} color="#f59e0b" />
          </div>
        </Card>

        {/* Nodes table */}
        <Card title="Nodes">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>ID</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Label</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Type</th>
                </tr>
              </thead>
              <tbody>
                {nodes.map((n, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12, color: '#475569' }}>{n.id}</td>
                    <td style={{ padding: '8px 12px', color: '#1e293b' }}>{n.label}</td>
                    <td style={{ padding: '8px 12px' }}><TypeBadge type={n.type} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  const renderMermaid = () => (
    <>
      <Card title="Flowchart Mermaid Definitions">
        <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
          Click "Load Definition" to fetch the Mermaid code for each flowchart.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {flowcharts.map((f, i) => {
            const isLoaded = detail && selectedId === f.id
            return (
              <div key={f.id || i} style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: 16, background: '#fafbfc' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{f.title}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{f.category} &middot; <ComplexityBadge complexity={f.complexity} /></div>
                  </div>
                  <button
                    onClick={() => loadDetail(f.id)}
                    style={{
                      padding: '6px 14px', borderRadius: 6, border: '1px solid #3b82f6',
                      background: isLoaded ? '#3b82f6' : '#fff', color: isLoaded ? '#fff' : '#3b82f6',
                      cursor: 'pointer', fontSize: 12, fontWeight: 600
                    }}
                  >{isLoaded ? 'Loaded' : 'Load Definition'}</button>
                </div>
                {isLoaded && detail.mermaid ? (
                  <pre style={{
                    background: '#1e293b', color: '#e2e8f0', padding: 16, borderRadius: 8,
                    fontSize: 12, overflowX: 'auto', whiteSpace: 'pre-wrap', lineHeight: 1.5, margin: 0
                  }}>{detail.mermaid}</pre>
                ) : isLoaded ? (
                  <div style={{ color: '#94a3b8', fontSize: 12, fontStyle: 'italic' }}>No mermaid definition available.</div>
                ) : null}
              </div>
            )
          })}
        </div>
      </Card>
    </>
  )

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Clinical Process Flowcharts
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Visual clinical process flows: node counts, decision points, complexity analysis, and Mermaid definitions
      </p>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
              background: tab === t.id ? '#1e293b' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#64748b'
            }}
          >{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'analytics' && renderAnalytics()}
      {tab === 'detail' && renderDetail()}
      {tab === 'mermaid' && renderMermaid()}
    </div>
  )
}
