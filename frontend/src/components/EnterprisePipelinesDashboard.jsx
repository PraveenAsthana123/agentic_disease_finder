import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b', '#8bc34a', '#ff5722', '#009688', '#795548']
const STATUS_COLORS = { built: '#4caf50', partial: '#ff9800', planned: '#f44336' }

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

function StagePills({ stages }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
      {stages.map((s, i) => (
        <span key={i} style={{
          background: '#f1f5f9', color: '#475569', borderRadius: 12,
          padding: '2px 8px', fontSize: 10, fontWeight: 500, border: '1px solid #e2e8f0'
        }}>
          {s}
        </span>
      ))}
    </div>
  )
}

export default function EnterprisePipelinesDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/enterprise-pipelines/overview`),
      axios.get(`${API_URL}/api/enterprise-pipelines/breakdown`),
      axios.get(`${API_URL}/api/enterprise-pipelines/definitions`),
    ])
      .then(([o, b, d]) => { setOverview(o.data); setBreakdown(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Enterprise Pipelines...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>Enterprise Pipelines data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'groups', label: 'By Group' },
    { id: 'all', label: 'All Pipelines' },
    { id: 'stages', label: 'Stage Map' },
    { id: 'definitions', label: 'Definitions' },
  ]
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Enterprise AI Control-Tower Pipelines</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
        {s.total_groups} groups, {s.total_pipelines} pipelines, {s.total_stages} stages, {s.built_pct}% built
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontSize: 12, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'groups' && <GroupsTab breakdown={breakdown} />}
      {tab === 'all' && <AllPipelinesTab breakdown={breakdown} />}
      {tab === 'stages' && <StageMapTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const s = overview.summary || {}
  const statusDist = overview.status_distribution || []
  const perGroup = overview.pipelines_per_group || []
  const stagesPerGroup = overview.stages_per_group || []

  return (
    <>
      {/* KPIs */}
      <Card>
        <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-around' }}>
          <KPI label="Groups" value={s.total_groups} />
          <KPI label="Pipelines" value={s.total_pipelines} />
          <KPI label="Total Stages" value={s.total_stages} />
          <KPI label="Avg Stages" value={s.avg_stages} sub="per pipeline" />
          <KPI label="Built" value={s.built} sub={`${s.built_pct}%`} />
          <KPI label="Partial" value={s.partial} />
          <KPI label="Planned" value={s.planned} />
          <KPI label="With Dashboard" value={s.has_maps_to} sub="linked" />
        </div>
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Status distribution pie */}
        <Card title="Pipeline Status Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={statusDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                {statusDist.map((_, i) => (
                  <Cell key={i} fill={STATUS_COLORS[statusDist[i].name] || COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        {/* Pipelines per group bar */}
        <Card title="Pipelines per Group">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={perGroup} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={75} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="built" stackId="a" fill="#4caf50" name="Built" />
              <Bar dataKey="partial" stackId="a" fill="#ff9800" name="Partial" />
              <Bar dataKey="planned" stackId="a" fill="#f44336" name="Planned" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Stages per group bar */}
      <Card title="Total Stages per Group">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={stagesPerGroup} margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 9, angle: -30 }} height={60} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#1e88e5" name="Stages" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Group summary table */}
      <Card title="Group Summary">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Group', 'Pipelines', 'Built', 'Partial', 'Planned', 'Stages'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {perGroup.map((g, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.name}</td>
                  <td style={{ padding: '6px 10px' }}>{g.value}</td>
                  <td style={{ padding: '6px 10px', color: '#4caf50' }}>{g.built}</td>
                  <td style={{ padding: '6px 10px', color: '#ff9800' }}>{g.partial}</td>
                  <td style={{ padding: '6px 10px', color: '#f44336' }}>{g.planned}</td>
                  <td style={{ padding: '6px 10px' }}>{g.total_stages}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  )
}

function GroupsTab({ breakdown }) {
  const groups = breakdown?.groups || []
  return (
    <>
      {groups.map((g, gi) => (
        <Card key={gi} title={`${g.group} (${g.pipeline_count} pipelines)`}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Pipeline', 'Stages', 'Status', 'Maps To'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {g.pipelines.map((p, pi) => (
                <tr key={pi} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.name}</td>
                  <td style={{ padding: '6px 10px' }}><StagePills stages={p.stages} /></td>
                  <td style={{ padding: '6px 10px' }}><StatusBadge status={p.status} /></td>
                  <td style={{ padding: '6px 10px', fontSize: 10, color: '#64748b', maxWidth: 300, wordBreak: 'break-word' }}>
                    {p.maps_to || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}
    </>
  )
}

function AllPipelinesTab({ breakdown }) {
  const flat = breakdown?.flat_table || []
  return (
    <Card title={`All Pipelines (${flat.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Group', 'Pipeline', 'Stages', 'Count', 'Status', 'Maps To'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {flat.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 500, color: '#64748b' }}>{p.group}</td>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.name}</td>
                <td style={{ padding: '6px 10px' }}><StagePills stages={p.stages} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.stage_count}</td>
                <td style={{ padding: '6px 10px' }}><StatusBadge status={p.status} /></td>
                <td style={{ padding: '6px 10px', fontSize: 10, color: '#64748b', maxWidth: 250, wordBreak: 'break-word' }}>
                  {p.maps_to || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function StageMapTab({ breakdown }) {
  const groups = breakdown?.groups || []
  return (
    <>
      <Card title="Pipeline Stage Flow Map">
        <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
          Visual stage flow for each pipeline — stages execute left-to-right.
        </p>
        {groups.map((g, gi) => (
          <div key={gi} style={{ marginBottom: 20 }}>
            <h4 style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 8, borderBottom: '1px solid #e2e8f0', paddingBottom: 4 }}>
              {g.group}
            </h4>
            {g.pipelines.map((p, pi) => (
              <div key={pi} style={{ marginBottom: 12, display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <div style={{ minWidth: 160, fontWeight: 600, fontSize: 12, color: '#334155', paddingTop: 4 }}>
                  {p.name}
                  <span style={{ marginLeft: 6 }}><StatusBadge status={p.status} /></span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 2 }}>
                  {p.stages.map((stage, si) => (
                    <React.Fragment key={si}>
                      <span style={{
                        background: '#e0f2fe', color: '#0369a1', borderRadius: 6,
                        padding: '4px 10px', fontSize: 11, fontWeight: 500, border: '1px solid #bae6fd',
                        whiteSpace: 'nowrap'
                      }}>
                        {stage}
                      </span>
                      {si < p.stages.length - 1 && (
                        <span style={{ color: '#94a3b8', fontSize: 14, fontWeight: 700 }}>→</span>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}
      </Card>
    </>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs?.available) return <Card title="Definitions"><p>Not available.</p></Card>

  return (
    <>
      <Card title="Group Descriptions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, width: 200 }}>Group</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {(defs.group_descriptions || []).map((g, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.group}</td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{g.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Status Legend">
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
          {(defs.status_legend || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <StatusBadge status={s.status} />
              <span style={{ fontSize: 12, color: '#475569' }}>{s.meaning}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Glossary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, width: 120 }}>Term</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {(defs.glossary || []).map((g, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{g.term}</td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Clinical Notes">
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
          {(defs.clinical_notes || []).map((n, i) => <li key={i} style={{ marginBottom: 4 }}>{n}</li>)}
        </ul>
      </Card>

      <Card title="References">
        <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
          {(defs.references || []).map((r, i) => <li key={i} style={{ marginBottom: 4 }}>{r}</li>)}
        </ol>
      </Card>
    </>
  )
}
