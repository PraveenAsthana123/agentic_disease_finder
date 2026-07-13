import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STATUS_COLORS = { built: '#10b981', planned: '#3b82f6', partial: '#f59e0b', scaffold: '#f59e0b' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 700, color: '#1e293b' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center', minWidth: 100 }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
    </div>
  )
}

function StatusBadge({ status }) {
  const c = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700, background: c + '18', color: c, border: `1px solid ${c}40` }}>
      {status}
    </span>
  )
}

export default function BmadDashboard() {
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
          axios.get(`${API_URL}/bmad/overview`),
          axios.get(`${API_URL}/bmad/breakdown`),
          axios.get(`${API_URL}/bmad/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load BMAD data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9881;</div>
      Loading BMAD Spec-Driven Agent data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  const tabs = ['overview', 'breakdown', 'definitions']
  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#475569',
    border: 'none', marginRight: 6
  })

  const statusDist = overview?.status_distribution || []
  const categoryDist = overview?.category_distribution || []
  const implList = overview?.implementation_completeness || []
  const needsSummary = overview?.needs_summary || []

  const perCategory = breakdown?.per_category || {}
  const builtDetail = breakdown?.built_agents_detail || []
  const plannedDetail = breakdown?.planned_agents_detail || []
  const moduleCov = breakdown?.module_coverage || {}
  const depMap = breakdown?.dependency_map || []
  const recentAct = breakdown?.recent_activity || []

  const defData = defs || {}

  return (
    <div style={{ padding: 20 }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, fontWeight: 800, color: '#1e293b' }}>
        BMAD — Spec-Driven Agent Development
      </h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        Breakthrough Method for Agile ai-agent Development — spec coverage, implementation status, dependency tracking
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} style={tabStyle(tab === t)} onClick={() => setTab(t)}>
            {t === 'overview' ? 'Overview' : t === 'breakdown' ? 'Agent Breakdown' : 'Methodology'}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', gap: 30, flexWrap: 'wrap', justifyContent: 'center' }}>
              <KPI label="Total Agents" value={overview?.total_agents} />
              <KPI label="Built" value={overview?.built_agents} color="#10b981" />
              <KPI label="Planned" value={overview?.planned_agents} color="#3b82f6" />
              <KPI label="Spec Coverage" value={overview?.spec_coverage_pct != null ? overview.spec_coverage_pct + '%' : '--'} color="#8b5cf6" />
              <KPI label="With Modules" value={overview?.agents_with_modules} />
              <KPI label="With Dependencies" value={overview?.agents_with_needs} />
              <KPI label="Clinical Decisions" value={overview?.total_decisions} color="#06b6d4" />
              <KPI label="Transactions" value={overview?.total_transactions} color="#64748b" />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 18 }}>
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={statusDist} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {statusDist.map((_, i) => (
                      <Cell key={i} fill={STATUS_COLORS[statusDist[i]?.status] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Category Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={categoryDist} layout="vertical" margin={{ left: 100 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="category" width={95} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Dependency Requirements">
            {needsSummary.length ? (
              <ResponsiveContainer width="100%" height={Math.max(180, needsSummary.length * 28)}>
                <BarChart data={needsSummary} layout="vertical" margin={{ left: 160 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="need" width={155} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No dependency data</div>}
          </Card>

          <Card title="Implementation Completeness">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Agent ID</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Task</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Module</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Deps</th>
                  </tr>
                </thead>
                <tbody>
                  {implList.slice(0, 25).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.agent_id}</td>
                      <td style={{ padding: '6px 10px' }}>{a.task}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={a.status} /></td>
                      <td style={{ padding: '6px 10px', color: a.has_module ? '#10b981' : '#ef4444' }}>{a.has_module ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '6px 10px', color: a.has_needs ? '#f59e0b' : '#94a3b8' }}>{a.has_needs ? 'Yes' : '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {implList.length > 25 && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 6 }}>Showing 25 of {implList.length} agents</div>}
            </div>
          </Card>
        </>
      )}

      {tab === 'breakdown' && (
        <>
          <Card title="Module Coverage">
            <div style={{ display: 'flex', gap: 30, flexWrap: 'wrap', justifyContent: 'center', marginBottom: 12 }}>
              <KPI label="Real Module" value={moduleCov.with_real_module} color="#10b981" />
              <KPI label="Planned Module" value={moduleCov.with_planned_module} color="#f59e0b" />
              <KPI label="Total" value={moduleCov.total} />
            </div>
          </Card>

          <Card title={`Built Agents (${builtDetail.length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>ID</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Task</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Module</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Needs</th>
                  </tr>
                </thead>
                <tbody>
                  {builtDetail.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.id}</td>
                      <td style={{ padding: '6px 10px' }}>{a.task}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 10, color: '#64748b' }}>{a.module}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#f59e0b' }}>{a.needs || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title={`Planned Agents (${plannedDetail.length})`}>
            {plannedDetail.length ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#fef3c7', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>ID</th>
                      <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Task</th>
                      <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Needs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {plannedDetail.map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.id}</td>
                        <td style={{ padding: '6px 10px' }}>{a.task}</td>
                        <td style={{ padding: '6px 10px', fontSize: 11, color: '#f59e0b' }}>{a.needs || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>All agents are built!</div>}
          </Card>

          <Card title="Dependency Map">
            {depMap.length ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Agent</th>
                      <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Dependencies</th>
                    </tr>
                  </thead>
                  <tbody>
                    {depMap.map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{d.agent_id}</td>
                        <td style={{ padding: '6px 10px', color: '#f59e0b' }}>{d.needs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No dependencies tracked</div>}
          </Card>

          <Card title="Recent Transaction Activity">
            {recentAct.length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={recentAct} margin={{ left: 80 }} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="action" width={75} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No transaction data</div>}
          </Card>

          {Object.keys(perCategory).length > 0 && (
            <Card title="Agents by Category">
              {Object.entries(perCategory).map(([cat, agents]) => (
                <div key={cat} style={{ marginBottom: 16 }}>
                  <h4 style={{ fontSize: 13, fontWeight: 700, color: '#3b82f6', margin: '0 0 6px' }}>{cat} ({agents.length})</h4>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {agents.map((a, i) => (
                      <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 10px', borderRadius: 6, fontSize: 11, background: '#f1f5f9' }}>
                        <span style={{ fontFamily: 'monospace' }}>{a.id}</span>
                        <StatusBadge status={a.status} />
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </Card>
          )}
        </>
      )}

      {tab === 'definitions' && (
        <>
          <Card title="BMAD Methodology">
            <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6, margin: 0 }}>
              {defData.bmad_method || 'Spec-driven agent methodology.'}
            </p>
          </Card>

          <Card title="Status Definitions">
            {defData.statuses && Object.entries(defData.statuses).map(([s, desc]) => (
              <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <StatusBadge status={s} />
                <span style={{ fontSize: 12, color: '#475569' }}>{desc}</span>
              </div>
            ))}
          </Card>

          <Card title="Spec Fields">
            {defData.spec_fields && (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Field</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(defData.spec_fields).map(([f, d]) => (
                    <tr key={f} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontWeight: 700 }}>{f}</td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{d}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          <Card title="Category Definitions">
            {defData.categories && Object.entries(defData.categories).map(([cat, desc]) => (
              <div key={cat} style={{ marginBottom: 8 }}>
                <span style={{ fontWeight: 700, fontSize: 12, color: '#1e293b' }}>{cat}:</span>
                <span style={{ fontSize: 12, color: '#475569', marginLeft: 6 }}>{desc}</span>
              </div>
            ))}
          </Card>

          <Card title="Compliance Note">
            <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6, margin: 0, fontStyle: 'italic' }}>
              {defData.compliance_note || '--'}
            </p>
          </Card>
        </>
      )}
    </div>
  )
}
