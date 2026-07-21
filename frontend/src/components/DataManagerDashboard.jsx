import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API = '/api'
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

const th = { padding: '6px 10px', textAlign: 'left', fontSize: 12, fontWeight: 600, color: '#475569', borderBottom: '2px solid #e2e8f0' }
const td = { padding: '6px 10px', fontSize: 12, color: '#334155', borderBottom: '1px solid #f1f5f9' }

export default function DataManagerDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/data-manager/overview`),
      axios.get(`${API}/data-manager/breakdown`),
      axios.get(`${API}/data-manager/definitions`),
    ]).then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setErr(e.message))
  }, [])

  if (err) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {err}</div>
  if (!ov) return <div style={{ padding: 24, color: '#64748b' }}>Loading Data Manager Dashboard...</div>

  const tabs = ['overview', 'tasks', 'dashboards', 'quality', 'definitions']
  const s = ov.summary || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, fontWeight: 700, color: '#0f172a' }}>
        Clinical Data Manager (CDM)
      </h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {ov.mission || 'Data-governance backbone of Responsible AI'}
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#3b82f6' : '#e2e8f0',
            color: tab === t ? '#fff' : '#475569',
            fontWeight: 600, fontSize: 13
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
          <Card title="KPIs" span={2}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'space-around' }}>
              <KPI label="Total Tasks" value={s.total_tasks} />
              <KPI label="Built" value={s.built} sub={`${s.built_pct}%`} />
              <KPI label="Total Steps" value={s.total_steps} />
              <KPI label="Challenges" value={s.total_challenges} />
              <KPI label="Sub-Dashboards" value={s.dashboards} sub={`${s.dashboards_built} built`} />
              <KPI label="Quality Assessments" value={s.quality_assessments} />
            </div>
          </Card>

          <Card title="Task Status Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {(ov.status_distribution || []).map((e, i) => (
                    <Cell key={i} fill={STATUS_COLORS[e.name] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Steps per Task">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(ov.task_table || []).map(t => ({ name: t.name, steps: t.steps_count }))}
                margin={{ left: 0, right: 10, top: 5, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" fontSize={10} interval={0} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="steps" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Tasks Summary" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={th}>Task</th>
                    <th style={th}>AI Feature</th>
                    <th style={th}>Deliverable</th>
                    <th style={th}>Steps</th>
                    <th style={th}>Challenges</th>
                    <th style={th}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.task_table || []).map((t, i) => (
                    <tr key={i}>
                      <td style={td}>{t.name}</td>
                      <td style={td}>{t.ai_feature}</td>
                      <td style={td}>{t.deliverable}</td>
                      <td style={{ ...td, textAlign: 'center' }}>{t.steps_count}</td>
                      <td style={{ ...td, textAlign: 'center' }}>{t.challenges_count}</td>
                      <td style={td}><StatusBadge status={t.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'tasks' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 16 }}>
          {(bd.per_task || []).map((t, i) => (
            <Card key={i} title={t.name}>
              <div style={{ display: 'flex', gap: 12, marginBottom: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                <StatusBadge status={t.status} />
                <span style={{ fontSize: 12, color: '#64748b' }}>AI: {t.ai_feature}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>Deliverable: {t.deliverable}</span>
              </div>
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Steps:</div>
                <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#334155' }}>
                  {(t.steps || []).map((s, j) => <li key={j} style={{ marginBottom: 3 }}>{s}</li>)}
                </ol>
              </div>
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Challenges:</div>
                <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#64748b' }}>
                  {(t.challenges || []).map((c, j) => <li key={j} style={{ marginBottom: 3 }}>{c}</li>)}
                </ul>
              </div>
              {t.endpoints && (Array.isArray(t.endpoints) ? t.endpoints : [t.endpoints]).filter(Boolean).length > 0 && (
                <div style={{ fontSize: 11, color: '#3b82f6' }}>
                  Endpoints: {(Array.isArray(t.endpoints) ? t.endpoints : [t.endpoints]).join(', ')}
                </div>
              )}
            </Card>
          ))}
        </div>
      )}

      {tab === 'dashboards' && (
        <Card title="Sub-Dashboards" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Dashboard</th>
                <th style={th}>Shows</th>
                <th style={th}>Endpoint</th>
                <th style={th}>Status</th>
              </tr>
            </thead>
            <tbody>
              {(ov.dashboard_table || []).map((d, i) => (
                <tr key={i}>
                  <td style={td}>{d.name}</td>
                  <td style={td}>{d.shows}</td>
                  <td style={{ ...td, fontSize: 11, color: '#3b82f6', fontFamily: 'monospace' }}>{d.endpoint || '—'}</td>
                  <td style={td}><StatusBadge status={d.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {tab === 'quality' && (
        <Card title="Quality Assessment Dimensions">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
            {(ov.quality_assessments || []).map((qa, i) => (
              <div key={i} style={{
                padding: '10px 14px', borderRadius: 6,
                background: '#f1f5f9', border: '1px solid #e2e8f0'
              }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{qa}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {tab === 'definitions' && df && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
          <Card title="Role Description" span={2}>
            <p style={{ fontSize: 13, color: '#334155', margin: 0, lineHeight: 1.6 }}>
              {df.role_description}
            </p>
          </Card>

          <Card title="Task Categories">
            {(df.task_categories || []).map((cat, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>{cat.name}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {(cat.tasks || []).map((t, j) => (
                    <span key={j} style={{
                      padding: '2px 8px', borderRadius: 4, fontSize: 11,
                      background: '#e0f2fe', color: '#0369a1', fontWeight: 500
                    }}>{t}</span>
                  ))}
                </div>
              </div>
            ))}
          </Card>

          <Card title="Status Legend">
            {(df.status_legend || []).map((s, i) => (
              <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 6, alignItems: 'center' }}>
                <StatusBadge status={s.status} />
                <span style={{ fontSize: 12, color: '#475569' }}>{s.description}</span>
              </div>
            ))}
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
              {(df.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: '6px 10px', background: '#f8fafc', borderRadius: 4, border: '1px solid #e2e8f0' }}>
                  <span style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{g.term}: </span>
                  <span style={{ fontSize: 12, color: '#475569' }}>{g.definition}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569' }}>
              {(df.clinical_notes || []).map((n, i) => <li key={i} style={{ marginBottom: 4 }}>{n}</li>)}
            </ul>
          </Card>

          <Card title="References">
            {(df.references || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: '#3b82f6' }}>{r.label}</span>
                <span style={{ fontSize: 11, color: '#64748b' }}> — {r.note}</span>
              </div>
            ))}
          </Card>
        </div>
      )}

      <div style={{ textAlign: 'right', fontSize: 10, color: '#94a3b8', marginTop: 16 }}>
        Source: config/data_manager.json | Updated: {ov.updated_at || '—'}
      </div>
    </div>
  )
}
