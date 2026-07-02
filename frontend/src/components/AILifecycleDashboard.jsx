import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API = '/api'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const STAGE_COLORS = {
  ideation: '#94a3b8', development: '#f59e0b', validation: '#8b5cf6',
  deployed: '#10b981', monitoring: '#3b82f6', retired: '#ef4444'
}
const PIE_COLORS = ['#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#06b6d4']

export default function AILifecycleDashboard() {
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
      axios.get(`${API}/ai-lifecycle/overview`),
      axios.get(`${API}/ai-lifecycle/breakdown`),
      axios.get(`${API}/ai-lifecycle/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'agents', label: 'Agent Lifecycle' },
    { id: 'pipelines', label: 'Pipeline Lifecycle' },
    { id: 'models', label: 'Model Inventory' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI Lifecycle dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No lifecycle data available.</div>

  const k = overview.kpis || {}

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Lifecycle Management</h2>
        <Badge
          text={`Coverage: ${k.lifecycle_coverage != null ? k.lifecycle_coverage + '%' : 'N/A'}`}
          color={k.lifecycle_coverage >= 70 ? '#10b981' : k.lifecycle_coverage >= 40 ? '#f59e0b' : '#ef4444'}
        />
        <Badge
          text={`Deployed: ${k.models_deployed ?? 0} models`}
          color="#3b82f6"
        />
        <Badge
          text={`Agents: ${k.agents_operational ?? 0} operational`}
          color="#10b981"
        />
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b', fontWeight: 600, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} k={k} />}
      {tab === 'agents' && <AgentsTab breakdown={breakdown} />}
      {tab === 'pipelines' && <PipelinesTab breakdown={breakdown} />}
      {tab === 'models' && <ModelsTab breakdown={breakdown} overview={overview} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview, k }) {
  const kpis = [
    { label: 'Total AI Assets', value: k.total_ai_assets, color: '#3b82f6' },
    { label: 'Lifecycle Coverage', value: (k.lifecycle_coverage ?? 0) + '%', color: k.lifecycle_coverage >= 70 ? '#10b981' : '#f59e0b' },
    { label: 'Models Deployed', value: k.models_deployed, color: '#8b5cf6' },
    { label: 'Agents Operational', value: k.agents_operational, color: '#10b981' },
    { label: 'Pipelines Active', value: k.pipelines_active, color: '#06b6d4' },
    { label: 'Validation Events', value: k.validation_events, color: '#f59e0b' },
    { label: 'Monitoring Events', value: k.monitoring_events, color: '#3b82f6' },
    { label: 'Training Runs', value: k.training_runs, color: '#8b5cf6' },
  ]

  const stageData = (overview.lifecycle_stage_distribution || []).map(s => ({
    name: s.stage, value: s.count
  }))

  const assetData = (overview.asset_type_distribution || []).map(a => ({
    name: a.type, value: a.count
  }))

  const dailyData = overview.daily_lifecycle_events || []

  const healthData = (overview.lifecycle_health || []).map(h => ({
    dimension: h.dimension, score: h.score
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((kp, i) => (
        <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
      ))}

      <Card title="Lifecycle Stage Distribution" span={2}>
        {stageData.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={stageData} cx="50%" cy="50%" outerRadius={90} dataKey="value" nameKey="name"
                label={({ name, value }) => `${name}: ${value}`}>
                {stageData.map((s, i) => (
                  <Cell key={i} fill={STAGE_COLORS[s.name] || PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No stage data</div>}
      </Card>

      <Card title="Asset Type Breakdown" span={2}>
        {assetData.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={assetData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                {assetData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No asset data</div>}
      </Card>

      <Card title="Daily Lifecycle Events" span={2}>
        {dailyData.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={dailyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No daily data</div>}
      </Card>

      <Card title="Lifecycle Health Radar" span={2}>
        {healthData.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={healthData} cx="50%" cy="50%" outerRadius={80}>
              <PolarGrid />
              <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} domain={[0, 100]} />
              <Radar dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No health data</div>}
      </Card>
    </div>
  )
}

function stageBadge(stage) {
  const color = STAGE_COLORS[stage] || '#64748b'
  return <Badge text={stage} color={color} />
}

function AgentsTab({ breakdown }) {
  const agents = breakdown?.agent_lifecycle || []
  const stageCounts = {}
  agents.forEach(a => { stageCounts[a.lifecycle_stage] = (stageCounts[a.lifecycle_stage] || 0) + 1 })
  const stageData = Object.entries(stageCounts).map(([stage, count]) => ({ stage, count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Agent Stage Distribution" span={2}>
        {stageData.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={stageData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="stage" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {stageData.map((s, i) => (
                  <Cell key={i} fill={STAGE_COLORS[s.stage] || PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No agents</div>}
      </Card>

      <Card title="Summary" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {Object.entries(stageCounts).map(([stage, count]) => (
            <div key={stage} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
              {stageBadge(stage)}
              <span style={{ fontWeight: 700, color: '#1e293b' }}>{count}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Agent Inventory" span={4}>
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Agent ID</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Status</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Lifecycle Stage</th>
              </tr>
            </thead>
            <tbody>
              {agents.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.id}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={a.status} color={a.status === 'built' ? '#10b981' : a.status === 'scaffold' ? '#f59e0b' : '#94a3b8'} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>{stageBadge(a.lifecycle_stage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function PipelinesTab({ breakdown }) {
  const pipelines = breakdown?.pipeline_lifecycle || []
  const groupCounts = {}
  pipelines.forEach(p => { groupCounts[p.group] = (groupCounts[p.group] || 0) + 1 })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Pipelines by Group" span={2}>
        {Object.keys(groupCounts).length ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={Object.entries(groupCounts).map(([g, c]) => ({ group: g, count: c }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="group" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" height={60} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No pipelines</div>}
      </Card>

      <Card title="Pipeline Status Mix" span={2}>
        {(() => {
          const statusCounts = {}
          pipelines.forEach(p => { statusCounts[p.lifecycle_stage] = (statusCounts[p.lifecycle_stage] || 0) + 1 })
          const data = Object.entries(statusCounts).map(([s, c]) => ({ name: s, value: c }))
          return data.length ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={data} cx="50%" cy="50%" outerRadius={90} dataKey="value" nameKey="name"
                  label={({ name, value }) => `${name}: ${value}`}>
                  {data.map((d, i) => (
                    <Cell key={i} fill={STAGE_COLORS[d.name] || PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>
        })()}
      </Card>

      <Card title="Pipeline Inventory" span={4}>
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Pipeline</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Group</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Stages</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Status</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Lifecycle</th>
              </tr>
            </thead>
            <tbody>
              {pipelines.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.name}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{p.group}</td>
                  <td style={{ padding: '6px 10px', fontSize: 10, color: '#94a3b8' }}>
                    {(p.stages || []).join(' > ')}
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={p.status} color={p.status === 'built' ? '#10b981' : p.status === 'partial' ? '#f59e0b' : '#94a3b8'} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>{stageBadge(p.lifecycle_stage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function ModelsTab({ breakdown, overview }) {
  const models = breakdown?.model_inventory || []
  const validationLog = breakdown?.validation_log || []
  const monitoringLog = breakdown?.monitoring_log || []
  const trainingHistory = breakdown?.training_history || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Deployed Model Inventory" span={4}>
        <div style={{ maxHeight: 300, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Model</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#64748b' }}>Size (KB)</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Last Modified</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Stage</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{m.name}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{m.file_size_kb}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11 }}>{m.last_modified}</td>
                  <td style={{ padding: '6px 10px' }}>{stageBadge(m.lifecycle_stage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Training History" span={2}>
        {trainingHistory.length ? (
          <div style={{ maxHeight: 250, overflow: 'auto' }}>
            {trainingHistory.map((t, i) => (
              <div key={i} style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                <span style={{ color: '#64748b' }}>{t.ts_utc || t.date}</span>
                <span style={{ marginLeft: 8 }}>{t.detail || t.action}</span>
              </div>
            ))}
          </div>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No training history</div>}
      </Card>

      <Card title="Validation Log" span={2}>
        {validationLog.length ? (
          <div style={{ maxHeight: 250, overflow: 'auto' }}>
            {validationLog.map((v, i) => (
              <div key={i} style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                <span style={{ color: '#64748b' }}>{v.created_at || v.date}</span>
                <span style={{ marginLeft: 8 }}>{v.type}: </span>
                <span style={{ fontWeight: 600 }}>{v.decision || v.finding || v.action}</span>
              </div>
            ))}
          </div>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No validation log</div>}
      </Card>

      <Card title="Monitoring Events" span={4}>
        {monitoringLog.length ? (
          <div style={{ maxHeight: 250, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: '#64748b' }}>Date</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: '#64748b' }}>Component</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: '#64748b' }}>Action</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: '#64748b' }}>Detail</th>
                </tr>
              </thead>
              <tbody>
                {monitoringLog.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '4px 10px', color: '#64748b', fontSize: 11 }}>{m.ts_utc || m.date}</td>
                    <td style={{ padding: '4px 10px' }}>{m.component}</td>
                    <td style={{ padding: '4px 10px' }}><Badge text={m.action} color="#3b82f6" /></td>
                    <td style={{ padding: '4px 10px', fontSize: 11, color: '#94a3b8', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No monitoring events</div>}
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  const sections = definitions?.sections || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {sections.map((sec, i) => (
        <Card key={i} title={sec.title}>
          <div style={{ display: 'grid', gap: 10 }}>
            {(sec.items || []).map((item, j) => (
              <div key={j} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
