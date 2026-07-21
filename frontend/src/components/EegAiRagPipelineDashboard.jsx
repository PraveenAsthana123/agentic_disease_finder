import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#94a3b8' }

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
  const key = (status || '').toLowerCase()
  const bg = STATUS_COLORS[key] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

function PhaseBadge({ phase }) {
  const idx = ['Data Acquisition', 'Preprocessing', 'Feature Engineering', 'Modeling', 'RAG Layer', 'Human Review & Output'].indexOf(phase)
  const color = COLORS[idx >= 0 ? idx : 0]
  return (
    <span style={{
      background: `${color}18`, color: color, border: `1px solid ${color}44`,
      borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 600
    }}>
      {phase}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function EegAiRagPipelineDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/eeg-ai-rag-pipeline/overview`),
      axios.get(`${API_URL}/eeg-ai-rag-pipeline/breakdown`),
      axios.get(`${API_URL}/eeg-ai-rag-pipeline/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading EEG AI RAG Pipeline...</div>

  const tabs = ['overview', 'by-phase', 'all-steps', 'flow', 'definitions']
  const kpis = overview.kpis || {}
  const phases = overview.phases || []
  const steps = overview.steps || []
  const statusDist = (overview.status_distribution || []).map(s => ({ name: s.status, value: s.count }))
  const stepsPerPhase = phases.map(p => ({ name: p.phase, value: p.total }))

  return (
    <div style={{ padding: '24px 32px', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>{overview.title}</h2>
        <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b' }}>{overview.note}</p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#475569'
          }}>
            {t === 'overview' ? 'Overview' : t === 'by-phase' ? 'By Phase' : t === 'all-steps' ? 'All Steps' : t === 'flow' ? 'Pipeline Flow' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Pipeline KPIs" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Steps" value={kpis.total_steps} />
              <KPI label="Built" value={kpis.built} sub="implemented" />
              <KPI label="Partial" value={kpis.partial} />
              <KPI label="Planned" value={kpis.planned} />
              <KPI label="Phases" value={kpis.phases} />
              <KPI label="Completion" value={`${kpis.completion_pct}%`} />
            </div>
          </Card>

          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={statusDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                  {statusDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Steps per Phase">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={stepsPerPhase}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-25} textAnchor="end" height={60} />
                <YAxis allowDecimals={false} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {stepsPerPhase.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Phase Completion" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={phases} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 10 }} />
                <YAxis dataKey="phase" type="category" tick={{ fontSize: 9 }} width={150} />
                <Tooltip />
                <Bar dataKey="built" name="Built" fill="#22c55e" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Pipeline Steps" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>Step</th>
                    <th style={thStyle}>Detail</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {steps.map(s => (
                    <tr key={s.n}>
                      <td style={tdStyle}>{s.n}</td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{s.step}</td>
                      <td style={{ ...tdStyle, fontSize: 11, color: '#64748b' }}>{s.detail}</td>
                      <td style={tdStyle}><StatusBadge status={s.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── By Phase Tab ── */}
      {tab === 'by-phase' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          {phases.map((phase, pi) => {
            const lo = [1, 5, 8, 14, 18, 21][pi]
            const hi = [4, 7, 13, 17, 20, 23][pi]
            const phaseSteps = steps.filter(s => s.n >= lo && s.n <= hi)
            return (
              <Card key={phase.phase} title={
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', background: COLORS[pi], display: 'inline-block' }} />
                  {phase.phase}
                  <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 400 }}>({phase.built}/{phase.total} built)</span>
                </span>
              }>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>#</th>
                      <th style={thStyle}>Step</th>
                      <th style={thStyle}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {phaseSteps.map(s => (
                      <tr key={s.n}>
                        <td style={tdStyle}>{s.n}</td>
                        <td style={tdStyle}>
                          <div style={{ fontWeight: 600, fontSize: 12 }}>{s.step}</div>
                          <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{s.detail}</div>
                          {s.where && <div style={{ fontSize: 10, color: '#3b82f6', marginTop: 2 }}>{s.where}</div>}
                        </td>
                        <td style={tdStyle}><StatusBadge status={s.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )
          })}
        </div>
      )}

      {/* ── All Steps Tab ── */}
      {tab === 'all-steps' && breakdown && (
        <Card title="Complete 23-Step Pipeline">
          <div style={{ maxHeight: 600, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>#</th>
                  <th style={thStyle}>Step</th>
                  <th style={thStyle}>Detail</th>
                  <th style={thStyle}>Phase</th>
                  <th style={thStyle}>Where</th>
                  <th style={thStyle}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.steps || []).map(s => (
                  <tr key={s.n}>
                    <td style={tdStyle}>{s.n}</td>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{s.step}</td>
                    <td style={{ ...tdStyle, fontSize: 11, color: '#64748b', maxWidth: 200 }}>{s.detail}</td>
                    <td style={tdStyle}><PhaseBadge phase={s.phase || ''} /></td>
                    <td style={{ ...tdStyle, fontSize: 10, color: '#3b82f6', maxWidth: 250, wordBreak: 'break-word' }}>{s.where || '\u2014'}</td>
                    <td style={tdStyle}><StatusBadge status={s.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Pipeline Flow Tab ── */}
      {tab === 'flow' && (
        <Card title="Pipeline Flow (Step-by-Step)">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 600, overflow: 'auto' }}>
            {phases.map((phase, pi) => {
              const lo = [1, 5, 8, 14, 18, 21][pi]
              const hi = [4, 7, 13, 17, 20, 23][pi]
              const phaseSteps = steps.filter(s => s.n >= lo && s.n <= hi)
              return (
                <React.Fragment key={phase.phase}>
                  <div style={{
                    background: `${COLORS[pi]}12`, border: `1px solid ${COLORS[pi]}33`,
                    borderRadius: 8, padding: 12, marginBottom: 4
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 13, color: COLORS[pi], marginBottom: 8 }}>
                      {phase.phase} ({phase.built}/{phase.total})
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {phaseSteps.map((s, si) => (
                        <React.Fragment key={s.n}>
                          <div style={{
                            background: '#fff', borderRadius: 6, padding: '6px 10px',
                            border: `1px solid ${s.status === 'built' ? '#22c55e' : '#e2e8f0'}`,
                            minWidth: 120, flex: '1 1 auto'
                          }}>
                            <div style={{ fontSize: 10, color: '#94a3b8' }}>Step {s.n}</div>
                            <div style={{ fontSize: 12, fontWeight: 600, color: '#1e293b' }}>{s.step}</div>
                            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{s.detail}</div>
                          </div>
                          {si < phaseSteps.length - 1 && (
                            <div style={{ display: 'flex', alignItems: 'center', color: '#cbd5e1', fontSize: 16 }}>\u2192</div>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                  {pi < phases.length - 1 && (
                    <div style={{ textAlign: 'center', color: '#cbd5e1', fontSize: 18 }}>\u2193</div>
                  )}
                </React.Fragment>
              )
            })}
          </div>
        </Card>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          <Card title="Phase Descriptions" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Phase</th>
                  <th style={thStyle}>Steps</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.phases || []).map((p, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: COLORS[i], marginRight: 6 }} />
                      {p.name}
                    </td>
                    <td style={tdStyle}>{p.steps}</td>
                    <td style={{ ...tdStyle, fontSize: 11, color: '#64748b' }}>{p.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ fontWeight: 700, fontSize: 12, color: '#1e293b' }}>{g.term}</span>
                  <span style={{ fontSize: 11, color: '#64748b', marginLeft: 6 }}>{g.definition}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 11, color: '#334155', marginBottom: 6 }}>{typeof r === 'string' ? r : `${r.label}: ${r.note}`}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
