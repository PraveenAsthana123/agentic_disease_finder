import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { automatic: '#4caf50', semi: '#ff9800', planned: '#94a3b8' }

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Badge({ status }) {
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

export default function AutomaticPipelinesDashboard() {
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
          axios.get(`${API_URL}/api/automatic-pipelines/overview`),
          axios.get(`${API_URL}/api/automatic-pipelines/breakdown`),
          axios.get(`${API_URL}/api/automatic-pipelines/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Automatic Pipelines data')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Automatic Pipelines...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#94a3b8' }}>Automatic Pipelines data not available.</div>

  const s = overview.summary || {}
  const tabs = ['overview', 'pipelines', 'stages', 'triggers', 'definitions']

  // Chart data
  const statusData = Object.entries(overview.status_distribution || {}).map(([name, value]) => ({ name, value }))
  const triggerData = Object.entries(overview.trigger_distribution || {}).map(([name, value]) => ({ name, value }))
  const endpointData = Object.entries(overview.endpoint_type_distribution || {}).map(([name, value]) => ({ name, value }))

  // Stage count per pipeline for bar chart
  const pipelineStages = (breakdown?.pipelines || []).map(p => ({
    name: p.process.length > 18 ? p.process.slice(0, 16) + '...' : p.process,
    stages: p.stage_count,
    status: p.status
  }))

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Automatic Pipelines Dashboard</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>End-to-end pipeline status, automation rate, and stage analytics</p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '1px solid #e2e8f0', paddingBottom: 8 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t ? '#1e293b' : 'transparent', color: tab === t ? '#fff' : '#64748b'
          }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          {/* KPI Grid */}
          <Card>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8 }}>
              <KPI label="Total Pipelines" value={fmt(s.total_pipelines)} />
              <KPI label="Automatic" value={fmt(s.automatic)} />
              <KPI label="Semi-auto" value={fmt(s.semi)} />
              <KPI label="Planned" value={fmt(s.planned)} />
              <KPI label="Automation Rate" value={`${fmt(s.automation_pct)}%`} />
              <KPI label="Total Stages" value={fmt(s.total_stages)} />
              <KPI label="Avg Stages" value={fmt(s.avg_stages_per_pipeline)} />
              <KPI label="Max Stages" value={fmt(s.max_stages)} />
            </div>
          </Card>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={statusData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {statusData.map((_, i) => <Cell key={i} fill={STATUS_COLORS[statusData[i].name] || COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Trigger Types">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={triggerData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {triggerData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Endpoint Types">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={endpointData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── Pipelines Tab ── */}
      {tab === 'pipelines' && (
        <Card title="All Pipelines">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Process</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Trigger</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Endpoint</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Stages</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.pipelines || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 500 }}>{p.process}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{p.trigger}</td>
                    <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 11, color: '#64748b' }}>{p.endpoint}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{p.stage_count}</td>
                    <td style={{ padding: '8px 6px' }}><Badge status={p.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Stages Tab ── */}
      {tab === 'stages' && (
        <>
          <Card title="Stages per Pipeline">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={pipelineStages} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="stages" radius={[0, 4, 4, 0]}>
                  {pipelineStages.map((entry, i) => (
                    <Cell key={i} fill={STATUS_COLORS[entry.status] || '#94a3b8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Stage Details">
            {(breakdown?.pipelines || []).map((p, i) => (
              <div key={i} style={{ marginBottom: 12, padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>{p.process}</span>
                  <Badge status={p.status} />
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {p.stages.map((stage, j) => (
                    <span key={j} style={{
                      background: '#e2e8f0', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#334155',
                      display: 'flex', alignItems: 'center', gap: 4
                    }}>
                      <span style={{ color: '#94a3b8', fontWeight: 600 }}>{j + 1}.</span> {stage}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </Card>
        </>
      )}

      {/* ── Triggers Tab ── */}
      {tab === 'triggers' && (
        <>
          <Card title="Trigger Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={triggerData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Pipelines by Trigger Type">
            {Object.entries(overview.trigger_distribution || {}).map(([trigger, count]) => {
              const matching = (breakdown?.pipelines || []).filter(p => {
                const tl = (p.trigger || '').toLowerCase()
                if (trigger === 'upload') return tl.includes('upload')
                if (trigger === 'scheduled') return tl.includes('cron') || tl.includes('scheduled')
                if (trigger === 'query') return tl.includes('query')
                if (trigger === 'stream') return tl.includes('stream')
                if (trigger === 'on-demand') return tl.includes('on-demand')
                return false
              })
              return (
                <div key={trigger} style={{ marginBottom: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, textTransform: 'capitalize' }}>
                    {trigger} ({count})
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {matching.map((p, i) => (
                      <span key={i} style={{
                        background: '#f1f5f9', borderRadius: 4, padding: '4px 10px', fontSize: 12, color: '#334155'
                      }}>
                        {p.process}
                      </span>
                    ))}
                  </div>
                </div>
              )
            })}
          </Card>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && (
        <Card title="Terminology">
          {(defs?.definitions || []).map((d, i) => (
            <div key={i} style={{ marginBottom: 10, padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{d.term}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{d.definition}</div>
            </div>
          ))}
        </Card>
      )}

      {/* Meta */}
      {breakdown?.meta && (
        <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8', textAlign: 'right' }}>
          Updated: {breakdown.meta.updated_at || '--'}
        </div>
      )}
    </div>
  )
}
