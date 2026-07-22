import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

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

function StatusBadge({ status }) {
  const colorMap = { fast: '#ef4444', moderate: '#eab308', slow: '#22c55e', high: '#ef4444', medium: '#eab308', low: '#22c55e' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function GNNElectrodeConnectivityDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/api/gnn-electrode-connectivity/overview`),
          axios.get(`${API_URL}/api/gnn-electrode-connectivity/breakdown`),
          axios.get(`${API_URL}/api/gnn-electrode-connectivity/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading GNN Electrode Connectivity data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'nodes', label: 'Node Features' },
    { id: 'edges', label: 'Edge Connectivity' },
    { id: 'patterns', label: 'Seizure Patterns' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const topAttention = overview?.top_attention_electrodes || []
  const spectralSummary = overview?.spectral_power_summary || []
  const regionSummary = overview?.region_summary || []
  const arch = overview?.model_architecture || {}

  /* --- Node Features data --- */
  const nodeFeatures = breakdown?.node_features || []
  const regionAttention = breakdown?.region_attention_summary || []

  /* --- Edge data --- */
  const edges = breakdown?.edges || []
  const regionConnectivity = breakdown?.region_connectivity || []
  const edgeDistribution = breakdown?.edge_distribution || []

  /* --- Seizure patterns data --- */
  const seizurePatterns = breakdown?.seizure_type_patterns || []
  const trainingConfig = breakdown?.training_config || {}

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>GNN Electrode Connectivity Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Graph Neural Network analysis of EEG electrode connectivity — spatial relationships, seizure propagation, and attention-based localization
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {/* ======================== OVERVIEW TAB ======================== */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Nodes (Electrodes)" value={fmt(overview.n_nodes)} color="#3b82f6" />
              <KPI label="Edges" value={fmt(overview.n_edges)} color="#8b5cf6" />
              <KPI label="Patients" value={fmt(overview.total_patients)} />
              <KPI label="Predictions" value={fmt(overview.total_predictions)} />
              <KPI label="Avg Coherence" value={fmt(overview.avg_coherence)} color="#22c55e" />
              <KPI label="Avg PLV" value={fmt(overview.avg_plv)} color="#06b6d4" />
            </div>
          </Card>

          {/* Top Attention Electrodes */}
          <Card title="Top Attention Electrodes" span={2}>
            <div style={{ maxHeight: 280, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Electrode</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Region</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Attention</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Neighbors</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Delta</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Theta</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Alpha</th>
                  </tr>
                </thead>
                <tbody>
                  {topAttention.map((n, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{n.electrode}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{n.region}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700, color: '#3b82f6' }}>{fmt(n.attention_weight)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{n.n_neighbors}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.delta_power)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.theta_power)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.alpha_power)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Spectral Power Distribution */}
          <Card title="Avg Spectral Power by Band">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={spectralSummary}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="band" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="avg_power" name="Avg Power" radius={[4, 4, 0, 0]}>
                  {spectralSummary.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Model Architecture */}
          <Card title="GNN Architecture" span={2}>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(arch).map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600, color: '#334155', textTransform: 'capitalize', width: 200 }}>
                      {k.replace(/_/g, ' ')}
                    </td>
                    <td style={{ padding: '6px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>
                      {String(v)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Strongest Connection */}
          {overview.strongest_connection && (
            <Card title="Strongest Connection">
              <div style={{ textAlign: 'center', padding: 16 }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#22c55e' }}>{overview.strongest_connection.pair}</div>
                <div style={{ fontSize: 13, color: '#64748b', marginTop: 6 }}>
                  Coherence: <strong>{fmt(overview.strongest_connection.coherence)}</strong>
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
                  Montage: {overview.montage}
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ======================== NODE FEATURES TAB ======================== */}
      {tab === 'nodes' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Full Node Table */}
          <Card title="All Electrode Node Features" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Electrode</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Region</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Delta</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Theta</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Alpha</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Beta</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Gamma</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Variance</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Kurtosis</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Attention</th>
                  </tr>
                </thead>
                <tbody>
                  {nodeFeatures.map((n, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{n.electrode}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{n.region}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.delta_power)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.theta_power)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.alpha_power)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.beta_power)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.gamma_power)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.variance)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(n.kurtosis)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700, color: '#3b82f6' }}>{fmt(n.attention_weight)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Region Attention Summary */}
          <Card title="Region Attention Summary">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={regionAttention} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="region" width={70} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg_attention" name="Avg Attention" radius={[0, 4, 4, 0]} fill="#3b82f6" />
                <Bar dataKey="max_attention" name="Max Attention" radius={[0, 4, 4, 0]} fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Attention by Electrode Bar Chart */}
          <Card title="Attention Weight by Electrode">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={nodeFeatures}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="electrode" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="attention_weight" name="Attention" radius={[4, 4, 0, 0]}>
                  {nodeFeatures.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== EDGE CONNECTIVITY TAB ======================== */}
      {tab === 'edges' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Edge Table */}
          <Card title="Electrode-Pair Edge Weights" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Source</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Target</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Coherence</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>PLV</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Mutual Info</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Granger</th>
                  </tr>
                </thead>
                <tbody>
                  {edges.map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{e.source}</td>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{e.target}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(e.coherence)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(e.plv)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(e.mutual_information)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(e.granger_causality)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Coherence Distribution */}
          <Card title="Edge Coherence Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={edgeDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="coherence_bin" tick={{ fontSize: 11 }} label={{ value: 'Coherence', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: 'Count', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Edge Count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Regional Connectivity Table */}
          <Card title="Regional Connectivity Matrix">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Region Pair</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Coherence</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg PLV</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Edges</th>
                  </tr>
                </thead>
                <tbody>
                  {regionConnectivity.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.region_pair}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.avg_coherence)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.avg_plv)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.n_edges}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== SEIZURE PATTERNS TAB ======================== */}
      {tab === 'patterns' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Seizure Type Graph Patterns Table */}
          <Card title="Seizure Type Graph Patterns" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Seizure Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Predictions</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Attention</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Dominant Region</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Propagation</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Lateralization</th>
                  </tr>
                </thead>
                <tbody>
                  {seizurePatterns.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.seizure_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.n_predictions)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700, color: '#3b82f6' }}>{fmt(p.avg_graph_attention)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.dominant_region}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><StatusBadge status={p.propagation_speed} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace' }}>{fmt(p.lateralization)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Predictions by Seizure Type Bar Chart */}
          <Card title="Predictions by Seizure Type">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={seizurePatterns} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="seizure_type" width={90} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="n_predictions" name="Predictions" radius={[0, 4, 4, 0]}>
                  {seizurePatterns.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Training Configuration */}
          <Card title="GNN Training Configuration">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(trainingConfig).filter(([k]) => k !== 'augmentation' && k !== 'note').map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600, color: '#334155', textTransform: 'capitalize', width: 200 }}>
                      {k.replace(/_/g, ' ')}
                    </td>
                    <td style={{ padding: '6px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>
                      {String(v)}
                    </td>
                  </tr>
                ))}
                {trainingConfig.augmentation && (
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600, color: '#334155', verticalAlign: 'top' }}>Augmentation</td>
                    <td style={{ padding: '6px 12px', color: '#475569', fontSize: 12 }}>
                      <ul style={{ margin: 0, paddingLeft: 16 }}>
                        {trainingConfig.augmentation.map((a, i) => <li key={i}>{a}</li>)}
                      </ul>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
            {trainingConfig.note && (
              <div style={{ marginTop: 12, padding: '8px 12px', background: '#fef3c7', borderRadius: 6, fontSize: 12, color: '#92400e' }}>
                {trainingConfig.note}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <Card title={definitions.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {(definitions.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 260 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}
