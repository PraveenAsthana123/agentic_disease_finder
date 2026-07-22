import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
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

const TABS = ['Overview', 'Channel Saliency', 'Attention Patterns', 'Diagnosis Comparison', 'Methodology']

export default function SaliencyAttentionDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/saliency-attention/overview`),
      axios.get(`${API}/api/saliency-attention/breakdown`),
      axios.get(`${API}/api/saliency-attention/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Saliency & Attention analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Saliency & Attention Map Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        EEG channel saliency and attention weight analysis — {fmt(kpis.total_analyses)} analyses, {fmt(kpis.n_channels)} channels
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

      {tab === 0 && <OverviewTab kpis={kpis} channelSaliency={ov.channel_saliency} temporalAttention={ov.temporal_attention} />}
      {tab === 1 && <ChannelSaliencyTab data={bd} />}
      {tab === 2 && <AttentionPatternsTab data={bd} />}
      {tab === 3 && <DiagnosisComparisonTab data={bd} />}
      {tab === 4 && <MethodologyTab definitions={df} />}
    </div>
  )
}

function OverviewTab({ kpis, channelSaliency, temporalAttention }) {
  const saliencyData = channelSaliency || []
  const temporalData = temporalAttention || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Total Analyses" value={fmt(kpis.total_analyses)} />
          <KPI label="Channels" value={fmt(kpis.n_channels)} />
          <KPI label="Top Salient Channel" value={kpis.top_salient_channel || '--'} color="#3b82f6" />
          <KPI label="Mean Attention Entropy" value={fmt(kpis.mean_attention_entropy)} color="#8b5cf6" />
        </div>
      </Card>

      <Card title="Channel Saliency Scores">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={saliencyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="saliency" name="Saliency Score" fill="#3b82f6" radius={[4, 4, 0, 0]}>
              {saliencyData.map((_, i) => (
                <React.Fragment key={i} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Temporal Attention Weights">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={temporalData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="segment" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="weight" name="Attention Weight" stroke="#3b82f6" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ChannelSaliencyTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const channelRanking = data.channel_ranking || []
  const chartData = channelRanking.map(c => ({
    channel: c.channel,
    score: c.score,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Channel Saliency Ranking">
        <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 28)}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="channel" width={80} />
            <Tooltip />
            <Bar dataKey="score" name="Saliency Score" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Channel Ranking Details">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Channel</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Score</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Rank</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>CI Lower</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>CI Upper</th>
              </tr>
            </thead>
            <tbody>
              {channelRanking.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{c.channel}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{fmt(c.score)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{fmt(c.rank)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{fmt(c.ci_lower)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{fmt(c.ci_upper)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function AttentionPatternsTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const attentionHeads = data.attention_heads || []
  const bandAttention = data.band_attention || []
  const temporalResolution = data.temporal_resolution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Attention Heads (Feature Weights)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={attentionHeads}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="feature" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="head_1" name="Head 1" fill={COLORS[0]} radius={[4, 4, 0, 0]} />
            <Bar dataKey="head_2" name="Head 2" fill={COLORS[1]} radius={[4, 4, 0, 0]} />
            <Bar dataKey="head_3" name="Head 3" fill={COLORS[2]} radius={[4, 4, 0, 0]} />
            <Bar dataKey="head_4" name="Head 4" fill={COLORS[3]} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Band Attention Weights">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={bandAttention}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="attention_weight" name="Attention Weight" fill="#3b82f6" radius={[4, 4, 0, 0]}>
              {bandAttention.map((_, i) => (
                <React.Fragment key={i} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Temporal Resolution">
        {temporalResolution.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Segment</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Start (s)</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>End (s)</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Weight</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Peak Channel</th>
                </tr>
              </thead>
              <tbody>
                {temporalResolution.map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{t.segment}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{fmt(t.start)}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{fmt(t.end)}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{fmt(t.weight)}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}>{t.peak_channel || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: 12 }}>No temporal resolution data available</div>
        )}
      </Card>
    </div>
  )
}

function DiagnosisComparisonTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const saliencyByDiagnosis = data.saliency_by_diagnosis || []

  /* Collect all unique channel keys across diagnoses for grouped bars */
  const channelKeys = []
  saliencyByDiagnosis.forEach(d => {
    Object.keys(d).forEach(k => {
      if (k !== 'diagnosis' && !channelKeys.includes(k)) channelKeys.push(k)
    })
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Top Channel Saliency by Diagnosis">
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={saliencyByDiagnosis}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="diagnosis" />
            <YAxis />
            <Tooltip />
            <Legend />
            {channelKeys.map((ch, i) => (
              <Bar key={ch} dataKey={ch} name={ch} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function MethodologyTab({ definitions }) {
  if (!definitions?.available) return <div style={{ color: '#f59e0b' }}>No definitions data</div>

  const methodology = definitions.methodology || []
  const clinicalRelevance = definitions.clinical_relevance || []
  const interpretationNotes = definitions.interpretation_notes || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {methodology.map((m, i) => (
        <Card key={i} title={m.name}>
          <p style={{ margin: '0 0 8px', fontSize: 13, color: '#475569' }}>{m.description}</p>
          {m.parameters && (
            <div style={{ margin: '8px 0', fontSize: 12, color: '#64748b' }}>
              <strong>Parameters:</strong> {m.parameters}
            </div>
          )}
          {m.strengths && m.limitations && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, margin: '12px 0' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#10b981', marginBottom: 4 }}>Strengths</div>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569' }}>
                  {m.strengths.map((s, j) => <li key={j} style={{ marginBottom: 2 }}>{s}</li>)}
                </ul>
              </div>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#ef4444', marginBottom: 4 }}>Limitations</div>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569' }}>
                  {m.limitations.map((l, j) => <li key={j} style={{ marginBottom: 2 }}>{l}</li>)}
                </ul>
              </div>
            </div>
          )}
          {m.reference && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{m.reference}</div>}
        </Card>
      ))}

      <Card title="Clinical Relevance">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {clinicalRelevance.map((cr, i) => <li key={i} style={{ marginBottom: 6 }}>{cr}</li>)}
        </ul>
      </Card>

      <Card title="Interpretation Notes">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {interpretationNotes.map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
        </ul>
      </Card>
    </div>
  )
}
