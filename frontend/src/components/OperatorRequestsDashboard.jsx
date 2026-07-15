import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4']
const STATUS_COLORS = {
  open: '#3b82f6', logged: '#94a3b8', addressed: '#10b981',
  pending: '#f59e0b', 'not-implemented': '#8b5cf6', rejected: '#ef4444'
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Request Details' },
  { id: 'definitions', label: 'Definitions' },
]

export default function OperatorRequestsDashboard() {
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
      axios.get(`${API_URL}/api/operator-requests/overview`),
      axios.get(`${API_URL}/api/operator-requests/breakdown`),
      axios.get(`${API_URL}/api/operator-requests/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Operator Requests data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Operator Requests Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Request lifecycle analytics — intake, triage, resolution, implementation tracking
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total Requests"><KPI label="All Time" value={data.total_requests} /></Card>
      <Card title="Actionable"><KPI label="Open + Pending" value={data.actionable} color="#f59e0b" /></Card>
      <Card title="Addressed"><KPI label="Completed" value={data.addressed} color="#10b981" /></Card>
      <Card title="Resolution Rate"><KPI label="Addressed / Total" value={`${data.resolution_rate}%`} color="#3b82f6" /></Card>

      <Card title="Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={data.status_distribution} dataKey="count" nameKey="status" cx="50%" cy="50%"
              outerRadius={90} label={({ status, count }) => `${status} (${count})`}>
              {data.status_distribution.map((e, i) => (
                <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Category Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.category_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="category" fontSize={12} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Source Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={data.source_distribution} dataKey="count" nameKey="source" cx="50%" cy="50%"
              outerRadius={80} label={({ source, count }) => `${source} (${count})`}>
              {data.source_distribution.map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Implementation Coverage" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginTop: 8 }}>
          <KPI label="Has Module" value={data.implementation_coverage.with_module}
            sub={`${data.implementation_coverage.module_pct}%`} color="#10b981" />
          <KPI label="Has API" value={data.implementation_coverage.with_api}
            sub={`${data.implementation_coverage.api_pct}%`} color="#3b82f6" />
          <KPI label="Tested" value={data.implementation_coverage.tested}
            sub={`${data.implementation_coverage.tested_pct}%`} color="#8b5cf6" />
        </div>
      </Card>

      {data.daily_volume && data.daily_volume.length > 0 && (
        <Card title="Daily Request Volume (21 days)" span={4}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.daily_volume}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" fontSize={11} tickFormatter={d => d?.slice(5)} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      {data.cross_tab && data.cross_tab.length > 0 && (
        <Card title="Category x Status" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Category</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Status</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>Count</th>
                </tr>
              </thead>
              <tbody>
                {data.cross_tab.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8 }}>{r.category}</td>
                    <td style={{ padding: 8 }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 4, fontSize: 12,
                        background: STATUS_COLORS[r.status] ? `${STATUS_COLORS[r.status]}18` : '#f1f5f9',
                        color: STATUS_COLORS[r.status] || '#64748b'
                      }}>{r.status}</span>
                    </td>
                    <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

function BreakdownTab({ data }) {
  const [catFilter, setCatFilter] = useState('all')
  const categories = Object.keys(data.per_category || {})

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        <Card title="Unaddressed"><KPI label="Open + Pending" value={data.unaddressed_count} color="#f59e0b" /></Card>
        <Card title="Implemented"><KPI label="With Module" value={data.implemented_count} color="#10b981" /></Card>
        <Card title="Categories"><KPI label="Distinct" value={categories.length} color="#3b82f6" /></Card>
      </div>

      <Card title="Unaddressed Requests" span={1}>
        <div style={{ maxHeight: 320, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>ID</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Request</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Category</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Source</th>
              </tr>
            </thead>
            <tbody>
              {(data.unaddressed || []).slice(0, 30).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, color: '#94a3b8' }}>#{r.id}</td>
                  <td style={{ padding: 8, maxWidth: 400 }}>{r.request_text}</td>
                  <td style={{ padding: 8 }}>{r.category}</td>
                  <td style={{ padding: 8, color: '#64748b' }}>{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Implemented Requests (with module tracking)">
        <div style={{ maxHeight: 320, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>ID</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Request</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Module</th>
                <th style={{ textAlign: 'left', padding: 8 }}>API</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Tested</th>
              </tr>
            </thead>
            <tbody>
              {(data.implemented || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, color: '#94a3b8' }}>#{r.id}</td>
                  <td style={{ padding: 8, maxWidth: 260 }}>{r.request_text}</td>
                  <td style={{ padding: 8, fontSize: 12, color: '#64748b' }}>{r.impl_module}</td>
                  <td style={{ padding: 8, fontSize: 12, fontFamily: 'monospace', color: '#3b82f6' }}>{r.impl_api}</td>
                  <td style={{ padding: 8 }}>
                    {r.tested ? <span style={{ color: '#10b981', fontWeight: 600 }}>{r.tested}</span> : <span style={{ color: '#94a3b8' }}>--</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Recent Requests (last 30)">
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>ID</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Request</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Category</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Status</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Source</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_requests || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, color: '#94a3b8' }}>#{r.id}</td>
                  <td style={{ padding: 8, maxWidth: 350 }}>{r.request_text}</td>
                  <td style={{ padding: 8 }}>{r.category}</td>
                  <td style={{ padding: 8 }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 4, fontSize: 12,
                      background: STATUS_COLORS[r.status] ? `${STATUS_COLORS[r.status]}18` : '#f1f5f9',
                      color: STATUS_COLORS[r.status] || '#64748b'
                    }}>{r.status}</span>
                  </td>
                  <td style={{ padding: 8, color: '#64748b' }}>{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  const sections = [
    { key: 'request_statuses', title: 'Request Statuses' },
    { key: 'categories', title: 'Categories' },
    { key: 'sources', title: 'Sources' },
    { key: 'implementation_fields', title: 'Implementation Fields' },
    { key: 'glossary', title: 'Glossary' },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map(s => {
        const items = data[s.key]
        if (!items) return null
        return (
          <Card key={s.key} title={s.title}>
            <dl style={{ margin: 0 }}>
              {Object.entries(items).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 10 }}>
                  <dt style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{k}</dt>
                  <dd style={{ margin: '2px 0 0 0', fontSize: 13, color: '#64748b' }}>{v}</dd>
                </div>
              ))}
            </dl>
          </Card>
        )
      })}
    </div>
  )
}
