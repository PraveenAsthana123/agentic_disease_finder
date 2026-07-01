import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Treemap
} from 'recharts'

const API = '/api'

const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4',
  '#e91e63', '#8bc34a', '#ff5722', '#607d8b', '#9c27b0', '#cddc39']

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

const TYPE_COLORS = {
  patient: '#1e88e5',
  disease: '#f44336',
  medication: '#4caf50',
  department: '#ff9800',
  analysis: '#7c4dff',
  mri_finding: '#00bcd4',
  neuropsych: '#e91e63',
  hitl_review: '#8bc34a',
  seizure_event: '#ff5722',
  document_type: '#607d8b',
}

export default function KnowledgeGraphDashboard() {
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
      axios.get(`${API}/knowledge-graph/overview`),
      axios.get(`${API}/knowledge-graph/breakdown`),
      axios.get(`${API}/knowledge-graph/definitions`),
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
    { id: 'entities', label: 'Entity Explorer' },
    { id: 'patients', label: 'Patient Subgraphs' },
    { id: 'network', label: 'Medication & Disease' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Knowledge Graph dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.note || 'No knowledge graph data available.'}</div>

  const s = overview.summary || {}

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Knowledge Graph AI</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        Entity-relationship graph from clinical.db — {s.total_nodes} nodes, {s.total_edges} edges, {s.entity_types} entity types
      </p>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#1e88e5' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(s, overview, breakdown)}
      {tab === 'entities' && renderEntities(overview, breakdown)}
      {tab === 'patients' && renderPatients(breakdown)}
      {tab === 'network' && renderNetwork(breakdown)}
      {tab === 'definitions' && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(s, overview, breakdown) {
  const typeData = (overview.type_distribution || []).map((t, i) => ({
    ...t, fill: TYPE_COLORS[t.type] || COLORS[i % COLORS.length]
  }))
  const relData = (overview.relation_distribution || []).map((r, i) => ({
    ...r, fill: COLORS[i % COLORS.length]
  }))
  const hubNodes = overview.hub_nodes || []
  const stats = breakdown?.graph_stats || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPIs */}
      <Card><KPI label="Total Nodes" value={s.total_nodes} color="#1e88e5" /></Card>
      <Card><KPI label="Total Edges" value={s.total_edges} color="#7c4dff" /></Card>
      <Card><KPI label="Entity Types" value={s.entity_types} color="#4caf50" /></Card>
      <Card><KPI label="Relation Types" value={s.relation_types} color="#ff9800" /></Card>
      <Card><KPI label="Avg Degree" value={s.avg_degree} sub="edges per node" color="#00bcd4" /></Card>
      <Card><KPI label="Max Degree" value={s.max_degree} sub="hub node" color="#f44336" /></Card>
      <Card><KPI label="Graph Density" value={s.density} sub="0 = sparse, 1 = full" color="#e91e63" /></Card>
      <Card><KPI label="Isolated Nodes" value={s.isolated_nodes} sub="no connections" color="#607d8b" /></Card>

      {/* Entity type distribution — pie chart */}
      <Card title="Entity Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={typeData} dataKey="count" nameKey="type" cx="50%" cy="50%"
              outerRadius={100} label={({ type, count }) => `${type} (${count})`}>
              {typeData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Relation type distribution — bar chart */}
      <Card title="Relationship Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={relData} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="relation" width={95} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" radius={[0, 4, 4, 0]}>
              {relData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Hub nodes table */}
      <Card title="Top Hub Nodes (Most Connected)" span={2}>
        <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Node ID</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Degree</th>
            </tr>
          </thead>
          <tbody>
            {hubNodes.map((h, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>{h.id}</td>
                <td style={{ padding: 8 }}>
                  <Badge text={h.type} color={TYPE_COLORS[h.type] || '#607d8b'} />
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>{h.degree}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Graph stats breakdown */}
      <Card title="Node & Edge Breakdown" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div>
            <h4 style={{ fontSize: 13, color: '#475569', margin: '0 0 8px' }}>Nodes by type</h4>
            {Object.entries(stats).filter(([k]) => k.startsWith('node_')).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 12 }}>
                <span style={{ color: '#64748b' }}>{k.replace('node_', '')}</span>
                <span style={{ fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </div>
          <div>
            <h4 style={{ fontSize: 13, color: '#475569', margin: '0 0 8px' }}>Edges by relation</h4>
            {Object.entries(stats).filter(([k]) => k.startsWith('edge_')).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 12 }}>
                <span style={{ color: '#64748b' }}>{k.replace('edge_', '')}</span>
                <span style={{ fontWeight: 600 }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  )
}

function renderEntities(overview, breakdown) {
  const nodes = breakdown?.nodes || []
  const edges = breakdown?.edges || []
  const typeGroups = {}
  nodes.forEach(n => {
    if (!typeGroups[n.type]) typeGroups[n.type] = []
    typeGroups[n.type].push(n)
  })

  // For treemap
  const treemapData = Object.entries(typeGroups).map(([type, items]) => ({
    name: type, size: items.length, fill: TYPE_COLORS[type] || '#607d8b'
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Entity Type Treemap" span={4}>
        <ResponsiveContainer width="100%" height={200}>
          <Treemap data={treemapData} dataKey="size" nameKey="name"
            stroke="#fff" content={({ x, y, width, height, name, size, fill }) => {
              if (width < 40 || height < 25) return null
              return (
                <g>
                  <rect x={x} y={y} width={width} height={height} fill={fill} stroke="#fff" strokeWidth={2} rx={4} />
                  <text x={x + width / 2} y={y + height / 2 - 6} textAnchor="middle" fill="#fff" fontSize={12} fontWeight={600}>{name}</text>
                  <text x={x + width / 2} y={y + height / 2 + 10} textAnchor="middle" fill="#ffffffcc" fontSize={11}>{size}</text>
                </g>
              )
            }} />
        </ResponsiveContainer>
      </Card>

      {/* Entity type cards */}
      {Object.entries(typeGroups).map(([type, items]) => (
        <Card key={type} title={`${type} (${items.length})`}>
          <div style={{ maxHeight: 200, overflowY: 'auto' }}>
            {items.slice(0, 15).map((n, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', fontSize: 12, borderBottom: '1px solid #f8fafc' }}>
                <span style={{ color: '#334155', fontFamily: 'monospace' }}>{n.label}</span>
                {n.predicted && <Badge text={n.predicted} color="#7c4dff" />}
                {n.age && <span style={{ color: '#94a3b8' }}>age {n.age}</span>}
                {n.gender && <span style={{ color: '#94a3b8' }}>{n.gender}</span>}
              </div>
            ))}
            {items.length > 15 && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>+{items.length - 15} more</div>}
          </div>
        </Card>
      ))}

      {/* Edge sample table */}
      <Card title={`Relationships (${edges.length} total, showing 50)`} span={4}>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Source</th>
                <th style={{ textAlign: 'center', padding: 6 }}>Relation</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Target</th>
              </tr>
            </thead>
            <tbody>
              {edges.slice(0, 50).map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontFamily: 'monospace', fontSize: 11 }}>{e.source}</td>
                  <td style={{ padding: 6, textAlign: 'center' }}>
                    <Badge text={e.relation} color="#1e88e5" />
                  </td>
                  <td style={{ padding: 6, fontFamily: 'monospace', fontSize: 11 }}>{e.target}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderPatients(breakdown) {
  const subgraphs = breakdown?.patient_subgraphs || []

  // Bar chart data — edges per patient
  const barData = subgraphs.map(p => ({
    patient: p.patient_id, edges: p.edges, neighbors: p.neighbors
  }))

  // Radar data — top 8 patients' relation breakdown
  const topPatients = subgraphs.slice(0, 8)
  const allRelTypes = [...new Set(topPatients.flatMap(p => Object.keys(p.relations || {})))]
  const radarData = allRelTypes.map(rel => {
    const entry = { relation: rel }
    topPatients.forEach(p => { entry[p.patient_id] = (p.relations || {})[rel] || 0 })
    return entry
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPIs */}
      <Card><KPI label="Total Patients" value={subgraphs.length} color="#1e88e5" /></Card>
      <Card><KPI label="Avg Edges / Patient" value={(subgraphs.reduce((a, p) => a + p.edges, 0) / (subgraphs.length || 1)).toFixed(1)} color="#7c4dff" /></Card>
      <Card><KPI label="Most Connected" value={subgraphs[0]?.patient_id || '-'} sub={`${subgraphs[0]?.edges || 0} edges`} color="#4caf50" /></Card>
      <Card><KPI label="Avg Neighbors" value={(subgraphs.reduce((a, p) => a + p.neighbors, 0) / (subgraphs.length || 1)).toFixed(1)} color="#ff9800" /></Card>

      {/* Edges per patient bar chart */}
      <Card title="Edges per Patient" span={4}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={barData} margin={{ bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="patient" angle={-45} textAnchor="end" tick={{ fontSize: 10 }} height={60} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="edges" fill="#1e88e5" radius={[4, 4, 0, 0]} name="Edges" />
            <Bar dataKey="neighbors" fill="#4caf50" radius={[4, 4, 0, 0]} name="Neighbors" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Radar chart — top patients by relation type */}
      {radarData.length > 0 && (
        <Card title="Top Patients — Relation Breakdown" span={4}>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="relation" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis tick={{ fontSize: 10 }} />
              {topPatients.slice(0, 4).map((p, i) => (
                <Radar key={p.patient_id} name={p.patient_id} dataKey={p.patient_id}
                  stroke={COLORS[i]} fill={COLORS[i]} fillOpacity={0.15} />
              ))}
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Patient subgraph table */}
      <Card title="Patient Subgraph Details" span={4}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Edges</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Neighbors</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Relations</th>
              </tr>
            </thead>
            <tbody>
              {subgraphs.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{p.edges}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{p.neighbors}</td>
                  <td style={{ padding: 8 }}>
                    {Object.entries(p.relations || {}).map(([rel, cnt]) => (
                      <span key={rel} style={{ marginRight: 8 }}>
                        <Badge text={`${rel}: ${cnt}`} color="#1e88e5" />
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderNetwork(breakdown) {
  const medications = breakdown?.medication_network || []
  const diseases = breakdown?.disease_clusters || []

  // Medication bar chart
  const medData = medications.map(m => ({ name: m.medication, patients: m.patients }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPIs */}
      <Card><KPI label="Medications" value={medications.length} color="#4caf50" /></Card>
      <Card><KPI label="Disease Clusters" value={diseases.length} color="#f44336" /></Card>
      <Card><KPI label="Most Prescribed" value={medications[0]?.medication || '-'} sub={`${medications[0]?.patients || 0} patients`} color="#1e88e5" /></Card>
      <Card><KPI label="Largest Cluster" value={diseases[0]?.disease || '-'} sub={`${diseases[0]?.patient_count || 0} patients`} color="#7c4dff" /></Card>

      {/* Medication bar chart */}
      <Card title="Medication Prescriptions" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={medData} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={95} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="patients" fill="#4caf50" radius={[0, 4, 4, 0]} name="Patients" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Disease clusters pie */}
      <Card title="Disease Clusters" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={diseases.map((d, i) => ({ ...d, name: d.disease, value: d.patient_count, fill: COLORS[i] }))}
              dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
              label={({ name, value }) => `${name} (${value})`}>
              {diseases.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Medication details table */}
      <Card title="Medication Details" span={4}>
        <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Medication</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Patients</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Dose (mg)</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Frequency</th>
            </tr>
          </thead>
          <tbody>
            {medications.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{m.medication}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>
                  <Badge text={m.patients} color="#4caf50" />
                </td>
                <td style={{ padding: 8, textAlign: 'right', color: '#64748b' }}>{m.dose_mg || '-'}</td>
                <td style={{ padding: 8, color: '#64748b' }}>{m.frequency || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Disease cluster details */}
      <Card title="Disease Cluster Details" span={4}>
        <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Patient Count</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Coverage</th>
            </tr>
          </thead>
          <tbody>
            {diseases.map((d, i) => {
              const totalPatients = diseases.reduce((a, x) => a + x.patient_count, 0)
              const pct = totalPatients > 0 ? ((d.patient_count / totalPatients) * 100).toFixed(1) : 0
              return (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{d.disease}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>
                    <Badge text={d.patient_count} color="#f44336" />
                  </td>
                  <td style={{ padding: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ flex: 1, height: 8, background: '#f1f5f9', borderRadius: 4 }}>
                        <div style={{ width: `${pct}%`, height: '100%', background: '#f44336', borderRadius: 4 }} />
                      </div>
                      <span style={{ fontSize: 12, color: '#64748b' }}>{pct}%</span>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function renderDefinitions(definitions) {
  if (!definitions?.available) return <div style={{ padding: 20, color: '#64748b' }}>No definitions available.</div>
  const metrics = definitions.metrics || []
  const entityTypes = definitions.entity_types || []
  const relationTypes = definitions.relation_types || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* Metrics */}
      <Card title="Graph Metrics" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {metrics.map((m, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{m.name}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{m.description}</div>
              {m.unit && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>Unit: {m.unit}</div>}
            </div>
          ))}
        </div>
      </Card>

      {/* Entity types */}
      <Card title="Entity Types" span={2}>
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Type</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Source</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {entityTypes.map((et, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}><Badge text={et.type} color={TYPE_COLORS[et.type] || '#607d8b'} /></td>
                <td style={{ padding: 6, fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>{et.source}</td>
                <td style={{ padding: 6, color: '#475569' }}>{et.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Relation types */}
      <Card title="Relation Types" span={2}>
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Relation</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {relationTypes.map((rt, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}><Badge text={rt.relation} color="#1e88e5" /></td>
                <td style={{ padding: 6, color: '#475569' }}>{rt.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Clinical relevance */}
      <Card title="Clinical Relevance" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[
            { title: 'Patient-Centric View', desc: 'Knowledge graphs enable holistic patient views connecting diagnoses, medications, test results, and clinical decisions in a single traversable structure.' },
            { title: 'Drug Interaction Detection', desc: 'Medication network analysis identifies polypharmacy patterns and potential drug-drug interactions across patient populations.' },
            { title: 'Disease Cohort Discovery', desc: 'Disease cluster analysis reveals patient groupings for cohort studies, treatment comparisons, and clinical trial eligibility.' },
            { title: 'Clinical Decision Support', desc: 'Graph traversal enables reasoning about patient similarity, treatment pathways, and outcome prediction based on connected clinical evidence.' },
            { title: 'IEC 62304 Traceability', desc: 'Entity-relationship tracking supports medical software lifecycle traceability requirements for regulatory submissions.' },
            { title: 'FDA AI/ML PCCP', desc: 'Knowledge graph provides transparency into data relationships and model input provenance for predetermined change control plans.' },
          ].map((item, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
