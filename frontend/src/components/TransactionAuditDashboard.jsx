import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Glossary' },
]

export default function TransactionAuditDashboard() {
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
      axios.get(`${API_URL}/api/transaction-audit/overview`),
      axios.get(`${API_URL}/api/transaction-audit/breakdown`),
      axios.get(`${API_URL}/api/transaction-audit/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Transaction Audit data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Transaction Audit Trail Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Audit trail analytics — transaction volume, component/action/actor distributions, human vs system activity
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

/* ─── Overview Tab ─── */
function OverviewTab({ data }) {
  const { total_transactions, top_components, action_distribution, actor_distribution, daily_volume, human_vs_system } = data

  const componentChart = (top_components || []).map(c => ({ name: c.component, count: c.count }))
  const actionChart = Object.entries(action_distribution || {}).slice(0, 10).map(([k, v]) => ({ name: k, count: v }))
  const actorChart = Object.entries(actor_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const volumeChart = (daily_volume || []).map(d => ({ date: d.date, count: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI row */}
      <Card>
        <KPI label="Total Transactions" value={total_transactions?.toLocaleString()} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="System Transactions" value={human_vs_system?.system?.toLocaleString()}
             sub={`${human_vs_system?.system_pct}%`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Human Transactions" value={human_vs_system?.human?.toLocaleString()}
             sub={`${human_vs_system?.human_pct}%`} color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="Components" value={Object.keys(data.component_distribution || {}).length}
             sub="distinct modules" color="#f59e0b" />
      </Card>

      {/* Daily volume */}
      <Card title="Daily Transaction Volume" span={4}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={volumeChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Top components */}
      <Card title="Top 10 Components by Volume" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={componentChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={100} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Action distribution */}
      <Card title="Top 10 Actions" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={actionChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={110} />
            <Tooltip />
            <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Actor distribution pie */}
      <Card title="Actor Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={actorChart} dataKey="value" nameKey="name" cx="50%" cy="50%"
                 outerRadius={90} label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
                 labelLine={{ stroke: '#94a3b8' }}>
              {actorChart.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Human vs System */}
      <Card title="Human vs System Breakdown" span={2}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, justifyContent: 'center', padding: 20 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ width: 120, height: 120, borderRadius: '50%', background: '#10b981',
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#fff' }}>{human_vs_system?.system_pct}%</div>
              <div style={{ fontSize: 10, color: '#d1fae5' }}>System</div>
            </div>
            <div style={{ marginTop: 8, fontSize: 13, color: '#334155' }}>{human_vs_system?.system?.toLocaleString()} txns</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ width: 120, height: 120, borderRadius: '50%', background: '#8b5cf6',
              display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#fff' }}>{human_vs_system?.human_pct}%</div>
              <div style={{ fontSize: 10, color: '#ede9fe' }}>Human</div>
            </div>
            <div style={{ marginTop: 8, fontSize: 13, color: '#334155' }}>{human_vs_system?.human?.toLocaleString()} txns</div>
          </div>
        </div>
      </Card>
    </div>
  )
}

/* ─── Breakdown Tab ─── */
function BreakdownTab({ data }) {
  const { per_component, per_actor, recent_transactions, hourly_pattern, patient_activity } = data
  const [expandedComp, setExpandedComp] = useState(null)

  const hourlyChart = (hourly_pattern || []).map(h => ({ hour: `${String(h.hour).padStart(2, '0')}:00`, count: h.count }))
  const patientChart = (patient_activity || []).map(p => ({ name: p.patient_id, count: p.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Hourly pattern */}
      <Card title="Hourly Activity Pattern (24h)" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={hourlyChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="hour" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={45} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Top patients */}
      <Card title="Top 20 Patients by Activity" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={patientChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 9 }} width={70} />
            <Tooltip />
            <Bar dataKey="count" fill="#ec4899" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Per-component accordion */}
      <Card title={`Component Detail (${(per_component || []).length} components)`} span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          {(per_component || []).map(c => (
            <div key={c.component} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <div onClick={() => setExpandedComp(expandedComp === c.component ? null : c.component)}
                   style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                     padding: '10px 8px', cursor: 'pointer', background: expandedComp === c.component ? '#f8fafc' : 'transparent' }}>
                <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{c.component}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>{c.total} transactions {expandedComp === c.component ? '\u25B2' : '\u25BC'}</span>
              </div>
              {expandedComp === c.component && (
                <div style={{ padding: '0 8px 12px', fontSize: 12, color: '#475569' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 4 }}>
                    <div>
                      <strong>Actions:</strong>
                      <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                        {Object.entries(c.actions || {}).map(([a, cnt]) => (
                          <li key={a}>{a}: {cnt}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <strong>Actors:</strong>
                      <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                        {Object.entries(c.actors || {}).map(([a, cnt]) => (
                          <li key={a}>{a}: {cnt}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8' }}>
                    Earliest: {c.earliest_ts || 'N/A'} | Latest: {c.latest_ts || 'N/A'}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Per-actor summary */}
      <Card title={`Actor Summary (${(per_actor || []).length} actors)`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#475569' }}>Actor</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#475569' }}>Transactions</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#475569' }}>Components</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#475569' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {(per_actor || []).map(a => (
                <tr key={a.actor} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 600, color: '#1e293b' }}>{a.actor}</td>
                  <td style={{ padding: '6px', textAlign: 'right', color: '#3b82f6', fontWeight: 600 }}>{a.total}</td>
                  <td style={{ padding: '6px', color: '#64748b' }}>{(a.components || []).join(', ')}</td>
                  <td style={{ padding: '6px', color: '#64748b' }}>{(a.actions || []).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Recent transactions */}
      <Card title="Recent 50 Transactions" span={2}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: '#475569' }}>ID</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: '#475569' }}>Patient</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: '#475569' }}>Component</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: '#475569' }}>Action</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: '#475569' }}>Actor</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: '#475569' }}>Detail</th>
                <th style={{ textAlign: 'left', padding: '6px 4px', color: '#475569' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {(recent_transactions || []).map(tx => (
                <tr key={tx.id} style={{ borderBottom: '1px solid #f8fafc' }}>
                  <td style={{ padding: '4px', color: '#94a3b8' }}>{tx.id}</td>
                  <td style={{ padding: '4px', color: '#1e293b', fontWeight: 500 }}>{tx.patient_id}</td>
                  <td style={{ padding: '4px' }}>
                    <span style={{ background: '#eff6ff', color: '#3b82f6', padding: '2px 6px',
                      borderRadius: 4, fontSize: 10, fontWeight: 500 }}>{tx.component}</span>
                  </td>
                  <td style={{ padding: '4px', color: '#475569' }}>{tx.action}</td>
                  <td style={{ padding: '4px' }}>
                    <span style={{ background: tx.actor === 'system' ? '#f0fdf4' : '#faf5ff',
                      color: tx.actor === 'system' ? '#16a34a' : '#7c3aed',
                      padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 500 }}>{tx.actor}</span>
                  </td>
                  <td style={{ padding: '4px', color: '#64748b', maxWidth: 200, overflow: 'hidden',
                    textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{tx.detail}</td>
                  <td style={{ padding: '4px', color: '#94a3b8', fontSize: 10, whiteSpace: 'nowrap' }}>{tx.ts_local}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ─── Definitions Tab ─── */
function DefinitionsTab({ data }) {
  const { glossary, component_descriptions, action_descriptions } = data

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Audit Trail Glossary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: '8px 6px', color: '#475569', width: 180 }}>Term</th>
              <th style={{ textAlign: 'left', padding: '8px 6px', color: '#475569' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {(glossary || []).map(g => (
              <tr key={g.term} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 600, color: '#1e293b' }}>{g.term}</td>
                <td style={{ padding: '6px', color: '#475569' }}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Component Descriptions">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {Object.entries(component_descriptions || {}).map(([comp, desc]) => (
            <div key={comp} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{comp}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Action Type Reference">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          {Object.entries(action_descriptions || {}).map(([action, desc]) => (
            <div key={action} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{action}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
