import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { built: '#4caf50', cataloged: '#1e88e5', planned: '#94a3b8', partial: '#ff9800' }

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

function Card({ title, children }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 10, padding: 18, border: '1px solid #334155' }}>
      {title && <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 10, fontWeight: 600 }}>{title}</div>}
      {children}
    </div>
  )
}

function KPI({ label, value }) {
  return (
    <div style={{ textAlign: 'center', padding: '10px 6px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9' }}>{fmt(value)}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  )
}

export default function AIDarkFactoryDashboard() {
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
          axios.get(`${API_URL}/dark-factory/overview`),
          axios.get(`${API_URL}/dark-factory/breakdown`),
          axios.get(`${API_URL}/dark-factory/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load AI Dark Factory data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading AI Dark Factory data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#f87171' }}>
      <strong>Error:</strong> {error}
    </div>
  )

  const tabs = ['overview', 'flow', 'tools', 'patterns', 'definitions']
  const kpis = overview?.kpis || {}
  const charts = overview?.charts || {}

  return (
    <div style={{ padding: '20px 24px', color: '#e2e8f0' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>AI Dark Factory</h2>
      <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>
        Autonomous Software Factory — reference architecture and tool catalog
      </p>

      {/* Tab nav */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontSize: 12, fontWeight: 600, textTransform: 'capitalize',
            background: tab === t ? '#3b82f6' : '#334155',
            color: tab === t ? '#fff' : '#94a3b8'
          }}>{t}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 10 }}>
            <Card><KPI label="Flow Stages" value={kpis.total_flow_stages} /></Card>
            <Card><KPI label="Stages Built" value={kpis.stages_built} /></Card>
            <Card><KPI label="Stages Cataloged" value={kpis.stages_cataloged} /></Card>
            <Card><KPI label="Stages Planned" value={kpis.stages_planned} /></Card>
            <Card><KPI label="Flow Completion %" value={`${kpis.flow_completion_pct}%`} /></Card>
            <Card><KPI label="Tools Cataloged" value={kpis.total_tools} /></Card>
            <Card><KPI label="Patterns Built" value={kpis.patterns_built} /></Card>
            <Card><KPI label="Planes Built" value={`${kpis.planes_built}/${kpis.total_planes}`} /></Card>
          </div>

          {/* Charts */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
            <Card title="Flow Stage Status">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={charts.flow_status_pie || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.flow_status_pie || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Tool Status">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={charts.tool_status_pie || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.tool_status_pie || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Tools by Category">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={charts.tool_category_bar || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </div>
      )}

      {tab === 'flow' && (
        <Card title="End-to-End Flow Stages">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>#</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Stage</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Tool</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Produces</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.flow_stages || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '8px 10px' }}>{s.n}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{s.stage}</td>
                    <td style={{ padding: '8px 10px' }}>{s.tool || '--'}</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{s.produces || '--'}</td>
                    <td style={{ padding: '8px 10px' }}><Badge status={s.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'tools' && (
        <Card title="Tool Catalog">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Tool</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Purpose</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.tool_catalog || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '8px 10px', color: '#7c4dff', fontWeight: 600 }}>{t.category}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{t.tool}</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{t.for || '--'}</td>
                    <td style={{ padding: '8px 10px' }}><Badge status={t.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'patterns' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Card title="Architectural Patterns">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {(breakdown?.patterns || []).map((p, i) => (
                <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: 14, border: '1px solid #334155' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, fontSize: 14 }}>{p.name.replace(/_/g, ' ')}</span>
                    <Badge status={p.status} />
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>{p.description}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>Best for: {p.best_for} | Failure mode: {p.failure_mode}</div>
                  {p.note && <div style={{ fontSize: 11, color: '#4caf50', marginTop: 4 }}>{p.note}</div>}
                </div>
              ))}
            </div>
          </Card>
          <Card title="Planes">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {(breakdown?.planes || []).map((p, i) => (
                <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: 12, border: '1px solid #334155' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                    <span style={{ fontWeight: 700, fontSize: 13 }}>{p.plane}</span>
                    {p.status && <Badge status={p.status} />}
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>Components: {(p.components || []).join(', ')}</div>
                  {p.note && <div style={{ fontSize: 11, color: '#4caf50', marginTop: 2 }}>{p.note}</div>}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && (
        <Card title="Definitions">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(defs || []).map((d, i) => (
              <div key={i} style={{ padding: '8px 12px', background: '#0f172a', borderRadius: 6, border: '1px solid #334155' }}>
                <span style={{ fontWeight: 700, color: '#f1f5f9', fontSize: 12 }}>{d.term}</span>
                <span style={{ color: '#94a3b8', fontSize: 12, marginLeft: 8 }}>— {d.definition}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
