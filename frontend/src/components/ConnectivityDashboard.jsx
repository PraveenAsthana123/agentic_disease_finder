import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(4)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

const TABS = ['Overview', 'Connectivity Matrix', 'Graph Metrics', 'Band Analysis', 'Methodology']

export default function ConnectivityDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/connectivity/overview`),
      axios.get(`${API}/api/connectivity/breakdown`),
      axios.get(`${API}/api/connectivity/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Connectivity analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Connectivity Analysis Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        EEG connectivity measures — coherence, PLV, correlation across {kpis.n_channels} virtual channels, {kpis.n_samples} samples
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#3b82f6' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} bandConn={ov.band_connectivity} matrix={ov.connectivity_matrix} />}
      {tab === 1 && <MatrixTab matrix={ov.connectivity_matrix} strongest={bd?.strongest_connections} weakest={bd?.weakest_connections} />}
      {tab === 2 && <GraphTab metrics={bd?.graph_metrics} />}
      {tab === 3 && <BandTab bandConn={ov.band_connectivity} perBand={bd?.per_band_detail} />}
      {tab === 4 && <MethodologyTab definitions={df} />}
    </div>
  )
}

function OverviewTab({ kpis, bandConn, matrix }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Samples" value={kpis.n_samples} />
          <KPI label="Virtual Channels" value={kpis.n_channels} sub="band-power derived" />
          <KPI label="Mean Connectivity" value={fmt(kpis.mean_connectivity)} color="#3b82f6" />
          <KPI label="Graph Density" value={fmt(kpis.graph_density)} color="#8b5cf6" />
          <KPI label="Clustering Coeff" value={fmt(kpis.clustering_coeff)} color="#10b981" />
          <KPI label="Avg Path Length" value={fmt(kpis.avg_path_length)} color="#f59e0b" />
        </div>
      </Card>

      <Card title="Band Connectivity Summary">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={bandConn} margin={{ left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
            <Tooltip formatter={v => fmt(v)} />
            <Legend />
            <Bar dataKey="mean_coherence" name="Coherence" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="mean_plv" name="PLV" fill="#10b981" radius={[4, 4, 0, 0]} />
            <Bar dataKey="mean_correlation" name="Correlation" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Connectivity Radar">
        <ResponsiveContainer width="100%" height={240}>
          <RadarChart data={bandConn}>
            <PolarGrid />
            <PolarAngleAxis dataKey="band" />
            <PolarRadiusAxis domain={[0, 1]} />
            <Radar name="Coherence" dataKey="mean_coherence" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
            <Radar name="PLV" dataKey="mean_plv" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
            <Tooltip formatter={v => fmt(v)} />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function MatrixTab({ matrix, strongest, weakest }) {
  const bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
  const colorScale = (v) => {
    const r = Math.round(255 * (1 - v))
    const g = Math.round(100 + 155 * v)
    const b = Math.round(255 * v)
    return `rgb(${r}, ${g}, ${b})`
  }

  // Build matrix grid
  const matrixMap = {}
  ;(matrix || []).forEach(m => { matrixMap[`${m.source}-${m.target}`] = m.value })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Connectivity Matrix (Correlation)" span={2}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr>
                <th style={{ padding: '8px 12px' }}></th>
                {bands.map(b => <th key={b} style={{ padding: '8px 12px', textTransform: 'capitalize' }}>{b}</th>)}
              </tr>
            </thead>
            <tbody>
              {bands.map(row => (
                <tr key={row}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, textTransform: 'capitalize' }}>{row}</td>
                  {bands.map(col => {
                    const v = row === col ? 1.0 : (matrixMap[`${row}-${col}`] ?? matrixMap[`${col}-${row}`] ?? 0)
                    return (
                      <td key={col} style={{
                        padding: '8px 16px', textAlign: 'center', fontFamily: 'monospace',
                        background: colorScale(Math.abs(v)), color: Math.abs(v) > 0.6 ? '#fff' : '#1e293b',
                        borderRadius: 4, fontWeight: 600
                      }}>{v.toFixed(2)}</td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Strongest Connections (Top 10)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Pair</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Coherence</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>PLV</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Correlation</th>
              </tr>
            </thead>
            <tbody>
              {(strongest || []).map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.source} ↔ {s.target}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6' }}>{fmt(s.coherence)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#10b981' }}>{fmt(s.plv)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#f59e0b' }}>{fmt(s.correlation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Weakest Connections">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Pair</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Coherence</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>PLV</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Correlation</th>
              </tr>
            </thead>
            <tbody>
              {(weakest || []).map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.source} ↔ {s.target}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6' }}>{fmt(s.coherence)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#10b981' }}>{fmt(s.plv)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#f59e0b' }}>{fmt(s.correlation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function GraphTab({ metrics }) {
  if (!metrics) return <div style={{ color: '#f59e0b' }}>No graph metrics available</div>

  const graphData = [
    { metric: 'Density', value: metrics.density, desc: 'Fraction of possible edges present (threshold > 0.3)', ideal: '0.3–0.7' },
    { metric: 'Clustering', value: metrics.clustering_coefficient, desc: 'How tightly connected neighborhoods are', ideal: '> 0.5 (brain networks)' },
    { metric: 'Avg Path Length', value: metrics.avg_path_length, desc: 'Mean shortest distance between node pairs', ideal: '< 3 (small-world)' },
    { metric: 'Small-World Index', value: metrics.small_world_index, desc: 'σ = C/C_rand ÷ L/L_rand — >1 is small-world', ideal: '> 1.0' },
    { metric: 'Modularity', value: metrics.modularity_estimate, desc: 'Community structure strength (greedy estimate)', ideal: '0.3–0.7' },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Graph-Theoretic Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
          <KPI label="Density" value={fmt(metrics.density)} color="#8b5cf6" />
          <KPI label="Clustering Coeff" value={fmt(metrics.clustering_coefficient)} color="#10b981" />
          <KPI label="Avg Path Length" value={fmt(metrics.avg_path_length)} color="#f59e0b" />
          <KPI label="Small-World σ" value={fmt(metrics.small_world_index)} color="#3b82f6" sub={metrics.small_world_index > 1 ? 'Small-world ✓' : 'Not small-world'} />
          <KPI label="Modularity" value={fmt(metrics.modularity_estimate)} color="#ec4899" />
        </div>
      </Card>

      <Card title="Graph Metrics Comparison" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={graphData} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="metric" width={100} />
            <Tooltip formatter={v => fmt(v)} />
            <Bar dataKey="value" name="Value" radius={[0, 4, 4, 0]}>
              {graphData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Metric Details" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Metric</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Value</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Description</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Ideal Range</th>
              </tr>
            </thead>
            <tbody>
              {graphData.map((g, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{g.metric}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: COLORS[i % COLORS.length] }}>{fmt(g.value)}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{g.desc}</td>
                  <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#64748b' }}>{g.ideal}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function BandTab({ bandConn, perBand }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Per-Band Connectivity Comparison" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={bandConn} margin={{ left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
            <Tooltip formatter={v => fmt(v)} />
            <Legend />
            <Bar dataKey="mean_coherence" name="Coherence" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="mean_plv" name="PLV" fill="#10b981" radius={[4, 4, 0, 0]} />
            <Bar dataKey="mean_correlation" name="Correlation" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {(perBand || []).map((b, bi) => (
        <Card key={bi} title={`${b.band.charAt(0).toUpperCase() + b.band.slice(1)} Band — Pair Details`}>
          <div style={{ overflowX: 'auto', maxHeight: 250, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Pair</th>
                  <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Coh</th>
                  <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>PLV</th>
                  <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Corr</th>
                </tr>
              </thead>
              <tbody>
                {(b.pairs || []).map((p, pi) => (
                  <tr key={pi} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.source} ↔ {p.target}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#3b82f6' }}>{fmt(p.coherence)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#10b981' }}>{fmt(p.plv)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#f59e0b' }}>{fmt(p.correlation)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ))}
    </div>
  )
}

function MethodologyTab({ definitions }) {
  if (!definitions) return <div style={{ color: '#f59e0b' }}>No definitions available</div>

  const methods = definitions.methods || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Connectivity Methods — Definitions & Clinical Relevance">
        {methods.map((m, i) => (
          <div key={i} style={{
            padding: 16, marginBottom: 12, background: '#f8fafc', borderRadius: 8,
            borderLeft: `4px solid ${COLORS[i % COLORS.length]}`
          }}>
            <h4 style={{ margin: '0 0 6px', fontSize: 15, color: '#1e293b' }}>{m.name}</h4>
            <p style={{ margin: '0 0 6px', fontSize: 13, color: '#475569' }}>{m.description}</p>
            {m.formula_note && (
              <p style={{ margin: '0 0 6px', fontSize: 12, fontFamily: 'monospace', color: '#64748b' }}>{m.formula_note}</p>
            )}
            <p style={{ margin: '0 0 6px', fontSize: 12, color: '#10b981' }}>
              <strong>Clinical relevance:</strong> {m.clinical_relevance}
            </p>
            {m.references && m.references.length > 0 && (
              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                <strong>References:</strong> {m.references.join('; ')}
              </div>
            )}
          </div>
        ))}
      </Card>
    </div>
  )
}
