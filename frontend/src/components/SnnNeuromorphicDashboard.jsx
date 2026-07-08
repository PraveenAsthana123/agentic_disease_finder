import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${v}%` : '--')
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'spike_patterns', label: 'Spike Patterns' },
  { id: 'power_analysis', label: 'Power Analysis' },
  { id: 'learning_curves', label: 'Learning Curves' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SnnNeuromorphicDashboard() {
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
      axios.get(`${API_URL}/api/snn-neuromorphic/overview`),
      axios.get(`${API_URL}/api/snn-neuromorphic/breakdown`),
      axios.get(`${API_URL}/api/snn-neuromorphic/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading SNN Neuromorphic Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>SNN Neuromorphic Computing Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Spiking Neural Network analysis — spike-timing-dependent plasticity, neuromorphic power efficiency, temporal coding
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#334155', fontWeight: tab === t.id ? 600 : 400,
            fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {!ov.available && tab !== 'definitions' && (
        <Card><p style={{ color: '#94a3b8' }}>{ov.reason || 'No data available'}</p></Card>
      )}

      {/* ─── OVERVIEW TAB ─── */}
      {tab === 'overview' && ov.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card>
            <KPI label="Patients Analyzed" value={fmt(ov?.kpis?.total_patients)} sub="Total EEG sessions" color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Seizures Detected" value={fmt(ov?.kpis?.seizures_detected)} sub="SNN classified events" color="#ef4444" />
          </Card>
          <Card>
            <KPI label="SNN Latency" value={fmt(ov?.kpis?.snn_latency_ms)} sub="ms inference time" color="#8b5cf6" />
          </Card>
          <Card>
            <KPI label="Power Budget" value={fmt(ov?.kpis?.power_budget_mw)} sub="mW neuromorphic chip" color="#f59e0b" />
          </Card>

          <Card>
            <KPI label="Avg Spike Rate" value={fmt(ov?.kpis?.avg_spike_rate_hz)} sub="Hz across electrodes" color="#10b981" />
          </Card>
          <Card>
            <KPI label="Coding Efficiency" value={pct(ov?.kpis?.temporal_coding_efficiency_pct)} sub="Temporal vs rate coding" color="#06b6d4" />
          </Card>
          <Card>
            <KPI label="STDP Epochs" value={fmt(ov?.kpis?.stdp_epochs)} sub="Learning iterations" color="#ec4899" />
          </Card>
          <Card>
            <KPI label="Accuracy" value={pct(ov?.kpis?.classification_accuracy_pct)} sub="SNN seizure accuracy" color="#334155" />
          </Card>

          {/* Model Comparison Bar Chart */}
          <Card title="Model Comparison: Power vs Latency" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov?.model_comparison || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="power_mw" name="Power (mW)" fill="#3b82f6" />
                <Bar yAxisId="right" dataKey="latency_ms" name="Latency (ms)" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Temporal Coding Efficiency Pie */}
          <Card title="Temporal Coding Efficiency" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={(ov?.temporal_coding_distribution || []).filter(d => d.value > 0)}
                  dataKey="value"
                  nameKey="category"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.temporal_coding_distribution || []).filter(d => d.value > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── SPIKE PATTERNS TAB ─── */}
      {tab === 'spike_patterns' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-patient spike pattern table */}
          <Card title="Per-Patient Spike Pattern Summary" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Patient</th>
                    <th style={{ padding: '8px 12px' }}>Spike Rate (Hz)</th>
                    <th style={{ padding: '8px 12px' }}>Burst Count</th>
                    <th style={{ padding: '8px 12px' }}>Interspike Interval (ms)</th>
                    <th style={{ padding: '8px 12px' }}>Dominant Pattern</th>
                    <th style={{ padding: '8px 12px' }}>Seizure Detected</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.per_patient || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(p.spike_rate_hz)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(p.burst_count)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(p.interspike_interval_ms)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                          background: p.dominant_pattern === 'ictal' ? '#fee2e2' : p.dominant_pattern === 'interictal' ? '#fef3c7' : '#dcfce7',
                          color: p.dominant_pattern === 'ictal' ? '#991b1b' : p.dominant_pattern === 'interictal' ? '#92400e' : '#166534',
                        }}>{p.dominant_pattern ?? '--'}</span>
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: p.seizure_detected ? '#ef4444' : '#10b981', fontWeight: 600 }}>
                        {p.seizure_detected ? 'Yes' : 'No'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Electrode-level spike rate bar chart */}
          <Card title="Electrode Spike Rates (Hz)" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={bd?.electrode_spike_rates || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="electrode" type="category" width={80} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="spike_rate_hz" name="Spike Rate (Hz)" fill="#8b5cf6">
                  {(bd?.electrode_spike_rates || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── POWER ANALYSIS TAB ─── */}
      {tab === 'power_analysis' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Power consumption comparison table */}
          <Card title="Power Consumption Comparison" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Architecture</th>
                    <th style={{ padding: '8px 12px' }}>Power (mW)</th>
                    <th style={{ padding: '8px 12px' }}>Latency (ms)</th>
                    <th style={{ padding: '8px 12px' }}>Accuracy (%)</th>
                    <th style={{ padding: '8px 12px' }}>Energy/Inf (µJ)</th>
                    <th style={{ padding: '8px 12px' }}>Hardware</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.power_comparison || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: row.architecture === 'SNN' ? '#f0f9ff' : undefined }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: row.architecture === 'SNN' ? '#1e40af' : '#334155' }}>{row.architecture}</td>
                      <td style={{ padding: '8px 12px', color: row.architecture === 'SNN' ? '#10b981' : '#475569', fontWeight: row.architecture === 'SNN' ? 600 : 400 }}>{fmt(row.power_mw)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(row.latency_ms)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(row.accuracy_pct)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(row.energy_per_inference_uj)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.hardware ?? '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Efficiency radar chart */}
          <Card title="Efficiency Radar: Architecture Comparison" span={2}>
            <ResponsiveContainer width="100%" height={340}>
              <RadarChart data={bd?.efficiency_radar || []}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                <Radar name="SNN" dataKey="snn" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.18} strokeWidth={2} />
                <Radar name="CNN" dataKey="cnn" stroke="#ef4444" fill="#ef4444" fillOpacity={0.18} strokeWidth={2} />
                <Radar name="LSTM" dataKey="lstm" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.18} strokeWidth={2} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── LEARNING CURVES TAB ─── */}
      {tab === 'learning_curves' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* STDP learning curve line chart */}
          <Card title="STDP Learning Curve" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={bd?.stdp_learning_curve || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="train_accuracy" name="Train Accuracy" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="val_accuracy" name="Val Accuracy" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="synaptic_weight_mean" name="Avg Synaptic Weight" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 2" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Seizure type classification bar chart */}
          <Card title="Seizure Type Classification" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={bd?.seizure_type_classification || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="seizure_type" tick={{ fontSize: 11 }} angle={-10} textAnchor="end" height={48} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="true_positives" name="True Positives" fill="#10b981" />
                <Bar dataKey="false_positives" name="False Positives" fill="#ef4444" />
                <Bar dataKey="false_negatives" name="False Negatives" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="SNN Neuromorphic Terminology" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px', width: 220 }}>Term</th>
                  <th style={{ padding: '8px 12px' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs?.terms || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{t.term}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{t.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
