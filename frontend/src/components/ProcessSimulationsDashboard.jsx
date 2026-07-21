import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const MODE_COLORS = { Auto: '#3b82f6', Manual: '#f97316' }
const LAYER_COLORS = { Data: '#3b82f6', Process: '#22c55e', Accuracy: '#8b5cf6', Reporting: '#f97316', Backend: '#14b8a6' }

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

function ModeBadge({ mode }) {
  const bg = MODE_COLORS[mode] || MODE_COLORS[mode?.charAt(0)?.toUpperCase() + mode?.slice(1)] || '#94a3b8'
  const label = mode?.charAt(0)?.toUpperCase() + mode?.slice(1)
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {label}
    </span>
  )
}

function LayerTag({ layer }) {
  const bg = LAYER_COLORS[layer] || LAYER_COLORS[layer?.charAt(0)?.toUpperCase() + layer?.slice(1)] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}18`, color: bg, border: `1px solid ${bg}44`,
      borderRadius: 4, padding: '2px 6px', fontSize: 10, fontWeight: 500, marginRight: 4, display: 'inline-block'
    }}>
      {layer}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function ProcessSimulationsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/process-simulations/overview`),
      axios.get(`${API_URL}/process-simulations/breakdown`),
      axios.get(`${API_URL}/process-simulations/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Process Simulations...</div>
  if (!overview?.available) return <div style={{ padding: 40, color: '#ef4444' }}>simulations.json not found</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'by-role', label: 'By Role' },
    { id: 'endpoints', label: 'Endpoint Map' },
    { id: 'actors', label: 'Actors' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
        Process Simulations Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        Per-role end-to-end process simulations — 7 roles, ordered pipeline steps, auto vs manual, endpoint mapping
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* OVERVIEW */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Roles" value={kpis.total_roles} />
              <KPI label="Total Steps" value={kpis.total_steps} />
              <KPI label="Auto Steps" value={kpis.auto_steps} sub="AI/Pipeline" />
              <KPI label="Manual Steps" value={kpis.manual_steps} sub="Human action" />
              <KPI label="Layers" value={kpis.unique_layers} />
              <KPI label="Actors" value={kpis.unique_actors} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Layer Distribution Pie */}
            <Card title="Layer Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.layer_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name} (${value})`}>
                    {(overview.layer_distribution || []).map((_, i) => (
                      <Cell key={i} fill={Object.values(LAYER_COLORS)[i] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Mode Distribution Pie */}
            <Card title="Auto vs Manual">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.mode_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name} (${value})`}>
                    {(overview.mode_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={MODE_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Steps per Role Bar */}
          <Card title="Steps per Role">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={overview.steps_per_role} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Roles Summary Table */}
          <Card title="All Roles Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Process</th>
                    <th style={thStyle}>Steps</th>
                    <th style={thStyle}>Auto</th>
                    <th style={thStyle}>Manual</th>
                    <th style={thStyle}>Layers</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.roles_table || []).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}><strong>{r.icon} {r.role}</strong></td>
                      <td style={tdStyle}>{r.process}</td>
                      <td style={tdStyle}>{r.steps}</td>
                      <td style={tdStyle}>{r.auto}</td>
                      <td style={tdStyle}>{r.manual}</td>
                      <td style={tdStyle}>{(r.layers || '').split(', ').map((l, j) => <LayerTag key={j} layer={l} />)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* BY ROLE — step-by-step process cards */}
      {tab === 'by-role' && breakdown?.roles && (
        <>
          {breakdown.roles.map((role, ri) => (
            <Card key={ri} title={`${role.icon} ${role.role} — ${role.process}`}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                {(role.layer_breakdown || []).map((lb, j) => (
                  <span key={j} style={{ fontSize: 11, color: '#475569' }}>
                    <LayerTag layer={lb.name} /> {lb.value}
                  </span>
                ))}
                <span style={{ margin: '0 4px', color: '#cbd5e1' }}>|</span>
                {(role.mode_breakdown || []).map((mb, j) => (
                  <span key={j} style={{ fontSize: 11, color: '#475569' }}>
                    <ModeBadge mode={mb.name.toLowerCase()} /> {mb.value}
                  </span>
                ))}
              </div>

              {/* Step pipeline */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                {(role.steps || []).map((s, si) => (
                  <div key={si} style={{
                    display: 'flex', alignItems: 'flex-start', gap: 12, padding: '10px 0',
                    borderBottom: si < role.steps.length - 1 ? '1px solid #f1f5f9' : 'none'
                  }}>
                    {/* Step number + connector */}
                    <div style={{
                      minWidth: 32, height: 32, borderRadius: '50%',
                      background: s.mode === 'auto' ? '#3b82f622' : '#f9731622',
                      color: s.mode === 'auto' ? '#3b82f6' : '#f97316',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 13, fontWeight: 700, flexShrink: 0
                    }}>
                      {s.step}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                        <LayerTag layer={s.layer?.charAt(0)?.toUpperCase() + s.layer?.slice(1)} />
                        <ModeBadge mode={s.mode} />
                        <span style={{ fontSize: 11, color: '#94a3b8' }}>{s.actor}</span>
                      </div>
                      <div style={{ fontSize: 12, color: '#1e293b', fontWeight: 500, marginBottom: 2 }}>
                        {s.process}
                      </div>
                      <div style={{ fontSize: 11, color: '#64748b' }}>
                        <strong>In:</strong> {s.input} &rarr; <strong>Out:</strong> {s.output}
                      </div>
                      {s.maps_to && (
                        <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2, fontFamily: 'monospace' }}>
                          maps_to: {s.maps_to}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}

      {/* ENDPOINT MAP */}
      {tab === 'endpoints' && breakdown?.endpoint_map && (
        <Card title="Endpoint Mapping — All Steps to System Components">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Role</th>
                  <th style={thStyle}>Step</th>
                  <th style={thStyle}>Mode</th>
                  <th style={thStyle}>Maps To</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.endpoint_map.map((ep, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}>{ep.role}</td>
                    <td style={tdStyle}>{ep.step}</td>
                    <td style={tdStyle}><ModeBadge mode={ep.mode} /></td>
                    <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: 11 }}>{ep.maps_to}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ACTORS */}
      {tab === 'actors' && overview?.actor_distribution && (
        <>
          <Card title="Actor Distribution — Who Does What">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.actor_distribution} layout="vertical" margin={{ left: 100, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={90} />
                <Tooltip />
                <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Actor Index">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Steps</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.actor_distribution.map((a, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}><strong>{a.name}</strong></td>
                      <td style={tdStyle}>{a.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* DEFINITIONS */}
      {tab === 'definitions' && defs?.available && (
        <>
          {/* Layer Legend */}
          <Card title="Layer Types">
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs.layer_legend || []).map((l, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <LayerTag layer={l.layer} />
                  <span style={{ fontSize: 12, color: '#475569' }}>{l.description}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Mode Legend */}
          <Card title="Execution Modes">
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs.mode_legend || []).map((m, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <ModeBadge mode={m.mode.toLowerCase()} />
                  <span style={{ fontSize: 12, color: '#475569' }}>{m.description}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Glossary */}
          <Card title="Glossary">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ fontSize: 12, color: '#334155' }}>
                  <strong>{g.term}</strong> — {g.definition}
                </div>
              ))}
            </div>
          </Card>

          {/* Clinical Notes */}
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{n}</li>
              ))}
            </ul>
          </Card>

          {/* References */}
          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>
                  <strong>[{r.id}]</strong> {r.title}
                </li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
