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

const TABS = ['Overview', 'Architecture Comparison', 'Layer Analysis', 'Training Curves', 'Methodology']

export default function HybridPipelineDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/hybrid-pipeline/overview`),
      axios.get(`${API}/api/hybrid-pipeline/breakdown`),
      axios.get(`${API}/api/hybrid-pipeline/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Hybrid Pipeline analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Hybrid CNN-LSTM / CNN-Transformer Pipeline</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Architecture comparison — {fmt(kpis.total_analyses)} analyses, best: {kpis.best_architecture || '--'}
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#3b82f6' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontSize: 13, fontWeight: tab === i ? 600 : 400
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <ComparisonTab ov={ov} bd={bd} />}
      {tab === 2 && <LayerTab bd={bd} />}
      {tab === 3 && <TrainingTab bd={bd} />}
      {tab === 4 && <MethodologyTab df={df} />}
    </div>
  )
}

function OverviewTab({ ov }) {
  const kpis = ov.kpis || {}
  const comparison = ov.architecture_comparison || []
  const featureImp = ov.feature_importance || []
  const confDist = ov.confidence_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Analyses" value={fmt(kpis.total_analyses)} color="#3b82f6" /></Card>
      <Card><KPI label="Mean Confidence" value={fmtPct(kpis.mean_confidence)} color="#10b981" /></Card>
      <Card><KPI label="Architectures" value={kpis.architectures_compared || 2} color="#8b5cf6" /></Card>
      <Card><KPI label="Best Architecture" value={kpis.best_architecture || '--'} color="#f59e0b" /></Card>

      <Card title="Architecture Performance Comparison" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={comparison}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="architecture" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Legend />
            <Bar dataKey="accuracy" fill={COLORS[0]} name="Accuracy" />
            <Bar dataKey="f1" fill={COLORS[1]} name="F1 Score" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Feature Importance (Top 10)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={featureImp.slice(0, 10)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={11} />
            <YAxis dataKey="feature" type="category" fontSize={10} width={120} />
            <Tooltip />
            <Bar dataKey="importance" fill={COLORS[4]}>
              {featureImp.slice(0, 10).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Confidence Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={confDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" fill={COLORS[6]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Per-Disease Performance" span={2}>
        {(ov.temporal_performance || []).length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>Disease</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>N</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>CNN-LSTM Acc</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>CNN-Transformer Acc</th>
                </tr>
              </thead>
              <tbody>
                {(ov.temporal_performance || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6 }}>{r.disease}</td>
                    <td style={{ textAlign: 'right', padding: 6 }}>{r.count}</td>
                    <td style={{ textAlign: 'right', padding: 6 }}>{fmtPct(r.cnn_lstm_acc)}</td>
                    <td style={{ textAlign: 'right', padding: 6 }}>{fmtPct(r.cnn_transformer_acc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No per-disease data</p>}
      </Card>
    </div>
  )
}

function ComparisonTab({ ov, bd }) {
  const comparison = ov.architecture_comparison || []
  const perPatient = bd?.per_patient || []
  const attention = bd?.attention_weights || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Architecture Metrics" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Architecture</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Accuracy</th>
                <th style={{ textAlign: 'right', padding: 8 }}>F1</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Latency (ms)</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Params (M)</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{r.architecture}</td>
                  <td style={{ textAlign: 'right', padding: 8 }}>{fmtPct(r.accuracy)}</td>
                  <td style={{ textAlign: 'right', padding: 8 }}>{fmtPct(r.f1)}</td>
                  <td style={{ textAlign: 'right', padding: 8 }}>{fmt(r.latency_ms)}</td>
                  <td style={{ textAlign: 'right', padding: 8 }}>{fmt(r.params_M)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Attention Head Weights (Band Power Distribution)">
        {attention.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={attention}>
              <PolarGrid />
              <PolarAngleAxis dataKey="band" fontSize={11} />
              <PolarRadiusAxis fontSize={10} />
              <Radar name="Weight" dataKey="weight" stroke={COLORS[1]} fill={COLORS[1]} fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No attention data</p>}
      </Card>

      <Card title="Per-Patient Architecture Assignment">
        {perPatient.length > 0 ? (
          <div style={{ maxHeight: 250, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 4 }}>Patient</th>
                  <th style={{ textAlign: 'left', padding: 4 }}>Best Arch</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>Confidence</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>Temporal</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>Spectral</th>
                </tr>
              </thead>
              <tbody>
                {perPatient.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f8fafc' }}>
                    <td style={{ padding: 4 }}>{r.patient_id}</td>
                    <td style={{ padding: 4 }}>{r.best_architecture}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{fmtPct(r.confidence)}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{fmt(r.temporal_score)}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{fmt(r.spectral_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No per-patient data</p>}
      </Card>
    </div>
  )
}

function LayerTab({ bd }) {
  const layers = bd?.layer_analysis || []
  const gates = bd?.lstm_gates || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="CNN Layer Activations (Feature Distributions)" span={2}>
        {layers.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={layers}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="layer" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Legend />
              <Bar dataKey="mean_activation" fill={COLORS[0]} name="Mean Activation" />
              <Bar dataKey="std_activation" fill={COLORS[1]} name="Std Activation" />
            </BarChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No layer data</p>}
      </Card>

      <Card title="LSTM Gate Activations" span={2}>
        {gates.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={gates}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="gate" fontSize={11} />
              <YAxis domain={[0, 1]} fontSize={11} />
              <Tooltip />
              <Bar dataKey="activation" fill={COLORS[4]}>
                {gates.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No gate data</p>}
      </Card>
    </div>
  )
}

function TrainingTab({ bd }) {
  const curves = bd?.training_curves || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Training Loss Curve" span={1}>
        {curves.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={curves}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="cnn_lstm_loss" stroke={COLORS[0]} name="CNN-LSTM" dot={false} />
              <Line type="monotone" dataKey="cnn_transformer_loss" stroke={COLORS[1]} name="CNN-Transformer" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No training data</p>}
      </Card>

      <Card title="Training Accuracy Curve" span={1}>
        {curves.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={curves}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" fontSize={11} />
              <YAxis domain={[0, 1]} fontSize={11} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="cnn_lstm_acc" stroke={COLORS[0]} name="CNN-LSTM" dot={false} />
              <Line type="monotone" dataKey="cnn_transformer_acc" stroke={COLORS[1]} name="CNN-Transformer" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No training data</p>}
      </Card>

      <Card title="Epoch Details" span={2}>
        {curves.length > 0 ? (
          <div style={{ maxHeight: 200, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ padding: 4 }}>Epoch</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>LSTM Loss</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>LSTM Acc</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>Transformer Loss</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>Transformer Acc</th>
                </tr>
              </thead>
              <tbody>
                {curves.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f8fafc' }}>
                    <td style={{ padding: 4, textAlign: 'center' }}>{r.epoch}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{fmt(r.cnn_lstm_loss)}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{fmtPct(r.cnn_lstm_acc)}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{fmt(r.cnn_transformer_loss)}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{fmtPct(r.cnn_transformer_acc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No training data</p>}
      </Card>
    </div>
  )
}

function MethodologyTab({ df }) {
  if (!df) return <p style={{ color: '#94a3b8' }}>No definitions available</p>

  const archs = df.architectures || []
  const refs = df.references || []
  const hyper = df.hyperparameters || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Architecture Definitions">
        {archs.map((a, i) => (
          <div key={i} style={{ marginBottom: 16, padding: 12, background: '#f8fafc', borderRadius: 8 }}>
            <h4 style={{ margin: '0 0 6px', fontSize: 14, color: '#1e293b' }}>{a.name}</h4>
            <p style={{ margin: '0 0 4px', fontSize: 12, color: '#475569' }}>{a.description}</p>
            <p style={{ margin: 0, fontSize: 11, color: '#64748b' }}><strong>Pipeline:</strong> {a.pipeline}</p>
            {a.best_for && <p style={{ margin: '4px 0 0', fontSize: 11, color: '#10b981' }}><strong>Best for:</strong> {a.best_for}</p>}
          </div>
        ))}
      </Card>

      <Card title="Hyperparameters">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {Object.entries(hyper).map(([arch, params]) => (
            <div key={arch} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <h4 style={{ margin: '0 0 8px', fontSize: 13 }}>{arch}</h4>
              {Object.entries(params || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, padding: '2px 0' }}>
                  <span style={{ color: '#64748b' }}>{k}</span>
                  <span style={{ fontWeight: 600 }}>{String(v)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </Card>

      <Card title="Clinical References">
        <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
          {refs.map((r, i) => (
            <li key={i} style={{ marginBottom: 6 }}>{r}</li>
          ))}
        </ol>
      </Card>
    </div>
  )
}
