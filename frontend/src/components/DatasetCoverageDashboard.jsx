import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { built: '#4caf50', scaffold: '#ff9800', planned: '#94a3b8', partial: '#ff9800' }

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

export default function DatasetCoverageDashboard() {
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
          axios.get(`${API_URL}/api/dataset-coverage/overview`),
          axios.get(`${API_URL}/api/dataset-coverage/breakdown`),
          axios.get(`${API_URL}/api/dataset-coverage/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Dataset Coverage data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading dataset coverage data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Dataset coverage data not available.'}
    </div>
  )

  const summary = overview.summary || {}
  const targetScale = overview.target_scale || {}
  const modStatusDist = overview.modality_status_distribution || {}
  const modTierDist = overview.modality_tier_distribution || {}
  const phaseDist = overview.phase_status_distribution || {}
  const aiDist = overview.ai_stream_distribution || {}
  const modalities = breakdown?.modalities || []
  const aiStreams = breakdown?.ai_streams || []
  const phases = breakdown?.phases || []
  const providerQuestions = breakdown?.provider_questions || []
  const definitions = defs?.definitions || []

  const modStatusPie = [
    { name: 'Built', value: modStatusDist.built || 0 },
    { name: 'Scaffold', value: modStatusDist.scaffold || 0 },
    { name: 'Planned', value: modStatusDist.planned || 0 },
  ].filter(d => d.value > 0)

  const tierBar = Object.entries(modTierDist).map(([k, v]) => ({
    name: k.replace('tier_', 'Tier '), value: v
  }))

  const phasePie = [
    { name: 'Built', value: phaseDist.built || 0 },
    { name: 'Partial', value: phaseDist.partial || 0 },
    { name: 'Planned', value: phaseDist.planned || 0 },
  ].filter(d => d.value > 0)

  const aiPie = [
    { name: 'Built', value: aiDist.built || 0 },
    { name: 'Scaffold', value: aiDist.scaffold || 0 },
    { name: 'Planned', value: aiDist.planned || 0 },
  ].filter(d => d.value > 0)

  const pieColors = ['#4caf50', '#ff9800', '#94a3b8']

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 130
  })
  const tabs = ['overview', 'modalities', 'ai streams', 'phases', 'definitions']
  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', borderRadius: '6px 6px 0 0', fontSize: 13, fontWeight: 600,
    background: tab === t ? '#1e88e5' : '#f1f5f9', color: tab === t ? '#fff' : '#475569',
    border: tab === t ? '1px solid #1e88e5' : '1px solid #e2e8f0', borderBottom: 'none'
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 22, color: '#1e293b' }}>Dataset Coverage Dashboard</h2>

      <div style={{ display: 'flex', gap: 4, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => <div key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</div>)}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {[
              { label: 'Total Modalities', value: summary.total_modalities, color: COLORS[3] },
              { label: 'Modalities Built', value: summary.modalities_built, color: COLORS[0] },
              { label: 'Modalities Scaffold', value: summary.modalities_scaffold, color: COLORS[1] },
              { label: 'Modalities Planned', value: summary.modalities_planned, color: COLORS[7] },
              { label: 'Coverage %', value: `${summary.coverage_pct}%`, color: COLORS[0] },
              { label: 'AI Streams', value: summary.ai_streams_total, color: COLORS[4] },
              { label: 'Phases Total', value: summary.phases_total, color: COLORS[5] },
              { label: 'Phases Built', value: summary.phases_built, color: COLORS[0] },
            ].map((k, i) => (
              <div key={i} style={kpiStyle(k.color)}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: k.color }}>{fmt(k.value)}</div>
              </div>
            ))}
          </div>

          {/* Target Scale */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 10px', fontSize: 14, color: '#334155' }}>Target Scale</h4>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {Object.entries(targetScale).map(([k, v]) => (
                <div key={k} style={{ background: '#f8fafc', borderRadius: 6, padding: '8px 14px' }}>
                  <div style={{ fontSize: 11, color: '#64748b' }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: '#1e293b' }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Research Question */}
          {overview.research_question && (
            <div style={{ ...cardStyle, background: '#f0f9ff', border: '1px solid #bae6fd' }}>
              <h4 style={{ margin: '0 0 6px', fontSize: 13, color: '#0369a1' }}>Novel Research Question</h4>
              <p style={{ margin: 0, fontSize: 13, color: '#0c4a6e' }}>{overview.research_question}</p>
            </div>
          )}

          {/* Charts */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>Modality Status</h4>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={modStatusPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({name,value})=>`${name}: ${value}`}>
                    {modStatusPie.map((_, i) => <Cell key={i} fill={pieColors[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>Modalities by Tier</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={tierBar}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="value" fill={COLORS[3]} radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>Phase Progress</h4>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={phasePie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({name,value})=>`${name}: ${value}`}>
                    {phasePie.map((_, i) => <Cell key={i} fill={pieColors[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>AI Streams</h4>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={aiPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({name,value})=>`${name}: ${value}`}>
                    {aiPie.map((_, i) => <Cell key={i} fill={pieColors[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {tab === 'modalities' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>All Modalities ({modalities.length})</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Code</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Name</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Measures</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>AI Potential</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Tier</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Phase</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {modalities.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontWeight: 600 }}>{m.code}</td>
                    <td style={{ padding: '6px 10px' }}>{m.name}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{m.measures}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{m.ai_potential}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{m.tier}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{m.phase}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><Badge status={m.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'ai streams' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>AI Streams ({aiStreams.length})</h4>
          {aiStreams.map((s, i) => (
            <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, marginBottom: 12, border: '1px solid #e2e8f0' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 13 }}>{s.id}</span>
                <Badge status={s.status.includes('built') ? 'built' : s.status.includes('scaffold') ? 'scaffold' : 'planned'} />
              </div>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}><strong>Input:</strong> {s.input}</div>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}><strong>Output:</strong> {s.output}</div>
              <div style={{ fontSize: 12, color: '#475569' }}><strong>Models:</strong> {s.models.join(', ')}</div>
            </div>
          ))}
        </div>
      )}

      {tab === 'phases' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>Implementation Phases</h4>
            {phases.map((p, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid #f1f5f9' }}>
                <div>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>Phase {p.phase}</span>
                  <span style={{ marginLeft: 10, fontSize: 12, color: '#64748b' }}>{p.name}</span>
                </div>
                <Badge status={p.status} />
              </div>
            ))}
          </div>

          {providerQuestions.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>Provider Due-Diligence Questions ({providerQuestions.length})</h4>
              <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
                {providerQuestions.map((q, i) => <li key={i} style={{ marginBottom: 4 }}>{q}</li>)}
              </ol>
            </div>
          )}
        </>
      )}

      {tab === 'definitions' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14 }}>Definitions ({definitions.length})</h4>
          {definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 12, paddingBottom: 10, borderBottom: '1px solid #f1f5f9' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d.term}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
