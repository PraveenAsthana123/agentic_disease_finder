import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, LineChart, Line, Legend, ScatterChart, Scatter,
  ZAxis
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#795548']
const METHOD_COLORS = { captum_ig: '#1e88e5', lime: '#4caf50', shap: '#ff9800' }

export default function XAIDashboard() {
  const [overview, setOverview] = useState(null)
  const [captum, setCaptum] = useState(null)
  const [limeData, setLimeData] = useState(null)
  const [shapData, setShapData] = useState(null)
  const [gradcamData, setGradcamData] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, df] = await Promise.all([
          axios.get(`${API_URL}/xai-dashboard/overview?seconds=10`),
          axios.get(`${API_URL}/xai-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load XAI data')
      }
      setLoading(false)
    }
    load()
  }, [])

  const loadTab = async (tab) => {
    setActiveTab(tab)
    try {
      if (tab === 'captum' && !captum) {
        const r = await axios.get(`${API_URL}/xai-dashboard/captum?seconds=10`)
        setCaptum(r.data)
      } else if (tab === 'lime' && !limeData) {
        const r = await axios.get(`${API_URL}/xai-dashboard/lime?seconds=10`)
        setLimeData(r.data)
      } else if (tab === 'shap' && !shapData) {
        const r = await axios.get(`${API_URL}/xai-dashboard/shap?seconds=10`)
        setShapData(r.data)
      } else if (tab === 'gradcam' && !gradcamData) {
        const r = await axios.get(`${API_URL}/xai-dashboard/gradcam?seconds=10`)
        setGradcamData(r.data)
      } else if (tab === 'comparison' && !comparison) {
        const r = await axios.get(`${API_URL}/xai-dashboard/comparison?seconds=10`)
        setComparison(r.data)
      }
    } catch (err) {
      setError(`Failed to load ${tab}: ${err.message}`)
    }
  }

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🧠</div>
      Running Captum + LIME + SHAP + Grad-CAM on real EEG features...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'No EDF data available for XAI analysis.'}
    </div>
  )

  const cardStyle = {
    background: '#ffffff', borderRadius: 12, padding: 18,
    boxShadow: '0 1px 4px rgba(0,0,0,0.07)', marginBottom: 16
  }
  const kpiStyle = (color) => ({
    background: '#ffffff', borderRadius: 10, padding: '14px 18px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.06)', borderLeft: `4px solid ${color}`,
    flex: '1 1 180px', minWidth: 160
  })

  const tabs = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'captum', label: '🔬 Captum' },
    { id: 'lime', label: '🍋 LIME' },
    { id: 'shap', label: '📐 SHAP' },
    { id: 'gradcam', label: '🔥 Grad-CAM' },
    { id: 'comparison', label: '⚖️ Comparison' },
  ]

  return (
    <div style={{ padding: '20px 24px', background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ ...cardStyle, background: 'linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%)', color: '#fff' }}>
        <h2 style={{ margin: 0, fontSize: 20 }}>🧠 Explainable AI Dashboard — Captum + LIME + SHAP + Grad-CAM</h2>
        <p style={{ margin: '6px 0 0', opacity: 0.85, fontSize: 13 }}>
          Real XAI analysis on {overview.data?.n_channels}-channel EEG features from {overview.data?.file} · EU AI Act Art. 86 compliant
        </p>
      </div>

      {/* KPI tiles */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
        <div style={kpiStyle('#1e88e5')}>
          <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase' }}>Model Accuracy</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{(overview.model.accuracy * 100).toFixed(1)}%</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>{overview.model.type}</div>
        </div>
        <div style={kpiStyle('#7c4dff')}>
          <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase' }}>XAI Methods</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{overview.methods.length}</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>Captum · LIME · SHAP · Grad-CAM</div>
        </div>
        <div style={kpiStyle('#4caf50')}>
          <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase' }}>Features</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{overview.data?.n_features}</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>{overview.data?.n_channels} channels × 8 features</div>
        </div>
        <div style={kpiStyle('#ff9800')}>
          <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase' }}>Epochs</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{overview.data?.n_epochs}</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>{overview.data?.epoch_duration_s}s each · {overview.data?.seconds_analyzed}s total</div>
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => loadTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', fontSize: 13,
              fontWeight: activeTab === t.id ? 700 : 400, cursor: 'pointer',
              background: activeTab === t.id ? '#1e3a5f' : '#e2e8f0',
              color: activeTab === t.id ? '#fff' : '#475569',
            }}>
            {t.label}
          </button>
        ))}
        <button onClick={() => setShowDefs(!showDefs)}
          style={{ marginLeft: 'auto', padding: '8px 14px', borderRadius: 8, border: '1px solid #cbd5e1',
            background: showDefs ? '#f1f5f9' : '#fff', color: '#475569', fontSize: 12, cursor: 'pointer' }}>
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {/* Definitions panel */}
      {showDefs && defs && (
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>📖 XAI Method Definitions</h3>
          {defs.methods?.map((m, i) => (
            <div key={i} style={{ marginBottom: 14, padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{m.name}</div>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{m.citation}</div>
              <div style={{ fontSize: 12, color: '#334155', marginBottom: 6 }}>{m.principle}</div>
              <div style={{ display: 'flex', gap: 20, fontSize: 11 }}>
                <div><strong style={{ color: '#16a34a' }}>Strengths:</strong> {m.strengths?.join(' · ')}</div>
              </div>
              <div style={{ fontSize: 11, marginTop: 4 }}><strong style={{ color: '#dc2626' }}>Limitations:</strong> {m.limitations?.join(' · ')}</div>
              <div style={{ fontSize: 11, marginTop: 4, color: '#1e40af' }}><strong>Clinical:</strong> {m.clinical_use}</div>
            </div>
          ))}
          {defs.eu_ai_act && (
            <div style={{ padding: 10, background: '#eff6ff', borderRadius: 8, border: '1px solid #bfdbfe', fontSize: 12 }}>
              <strong>🇪🇺 {defs.eu_ai_act.article}:</strong> {defs.eu_ai_act.requirement}
            </div>
          )}
        </div>
      )}

      {/* Overview tab */}
      {activeTab === 'overview' && (
        <div>
          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Top 10 Features by SHAP Importance</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={overview.top_features_shap} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={135} />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <Bar dataKey="importance" fill="#1e88e5" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Available XAI Methods</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 12 }}>
              {overview.methods?.map((m, i) => (
                <div key={i} style={{ padding: 14, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: COLORS[i] }}>{m.name}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                    Library: {m.library} · Type: {m.type} · Scope: {m.scope}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Captum tab */}
      {activeTab === 'captum' && (
        <div>
          {!captum ? (
            <div style={{ padding: 24, textAlign: 'center', color: '#64748b' }}>Loading Captum Integrated Gradients...</div>
          ) : !captum.available ? (
            <div style={{ padding: 20, color: '#92400e' }}>{captum.note}</div>
          ) : (
            <>
              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: '#1e293b' }}>🔬 Integrated Gradients — Top Feature Attributions</h3>
                <p style={{ margin: '0 0 12px', fontSize: 12, color: '#64748b' }}>{captum.integrated_gradients.description}</p>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={captum.integrated_gradients.top_features} layout="vertical" margin={{ left: 140 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={135} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="attribution" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Channel Importance (Integrated Gradients)</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={captum.integrated_gradients.per_channel}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="importance" fill="#7c4dff" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: '#1e293b' }}>Feature Ablation — Top Attributions</h3>
                <p style={{ margin: '0 0 12px', fontSize: 12, color: '#64748b' }}>{captum.feature_ablation.description}</p>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={captum.feature_ablation.top_features} layout="vertical" margin={{ left: 140 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={135} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="attribution" fill="#e91e63" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      )}

      {/* LIME tab */}
      {activeTab === 'lime' && (
        <div>
          {!limeData ? (
            <div style={{ padding: 24, textAlign: 'center', color: '#64748b' }}>Loading LIME explanations...</div>
          ) : !limeData.available ? (
            <div style={{ padding: 20, color: '#92400e' }}>{limeData.note}</div>
          ) : (
            <>
              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: '#1e293b' }}>🍋 LIME — Global Feature Importance (mean |weight|)</h3>
                <p style={{ margin: '0 0 12px', fontSize: 12, color: '#64748b' }}>{limeData.description}</p>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={limeData.global_feature_importance} layout="vertical" margin={{ left: 140 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={135} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="mean_abs_weight" fill="#4caf50" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Sample Explanations</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
                  {limeData.sample_explanations?.map((s, i) => (
                    <div key={i} style={{ padding: 12, background: '#f0fdf4', borderRadius: 8, border: '1px solid #bbf7d0' }}>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
                        Epoch {s.sample_idx} — Predicted: {s.predicted_class}
                        {s.predicted_class !== s.true_class && <span style={{ color: '#dc2626' }}> (true: {s.true_class})</span>}
                      </div>
                      {s.top_contributions?.map((c, j) => (
                        <div key={j} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, padding: '2px 0',
                          borderBottom: j < s.top_contributions.length - 1 ? '1px solid #dcfce7' : 'none' }}>
                          <span style={{ color: '#334155' }}>{c.feature}</span>
                          <span style={{ fontWeight: 600, color: c.weight > 0 ? '#16a34a' : '#dc2626' }}>
                            {c.weight > 0 ? '+' : ''}{c.weight.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* SHAP tab */}
      {activeTab === 'shap' && (
        <div>
          {!shapData ? (
            <div style={{ padding: 24, textAlign: 'center', color: '#64748b' }}>Loading SHAP values...</div>
          ) : !shapData.available ? (
            <div style={{ padding: 20, color: '#92400e' }}>{shapData.note}</div>
          ) : (
            <>
              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>📐 SHAP — Global Feature Importance (mean |SHAP|)</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={shapData.global_importance} layout="vertical" margin={{ left: 140 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={135} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="mean_abs_shap" fill="#ff9800" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                <div style={cardStyle}>
                  <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Channel-Level SHAP</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={shapData.per_channel}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ fontSize: 12 }} />
                      <Bar dataKey="mean_abs_shap" fill="#ff9800" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div style={cardStyle}>
                  <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Frequency Band SHAP</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={Object.entries(shapData.per_band || {}).map(([k, v]) => ({ name: k, value: v }))}
                        dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value.toFixed(4)}`}>
                        {Object.keys(shapData.per_band || {}).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Pie>
                      <Tooltip contentStyle={{ fontSize: 12 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Waterfall Samples (per-epoch SHAP contributions)</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
                  {shapData.waterfall_samples?.map((s, i) => (
                    <div key={i} style={{ padding: 12, background: '#fff7ed', borderRadius: 8, border: '1px solid #fed7aa' }}>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
                        Epoch {s.sample_idx} — {s.predicted_class}
                        <span style={{ fontSize: 11, color: '#64748b', marginLeft: 8 }}>base: {s.base_value?.toFixed(4)}</span>
                      </div>
                      {s.contributions?.map((c, j) => (
                        <div key={j} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, padding: '2px 0',
                          borderBottom: j < s.contributions.length - 1 ? '1px solid #ffedd5' : 'none' }}>
                          <span style={{ color: '#334155' }}>{c.feature}</span>
                          <span style={{ fontWeight: 600, fontFamily: 'monospace',
                            color: c.shap_value > 0 ? '#ea580c' : '#2563eb' }}>
                            {c.shap_value > 0 ? '+' : ''}{c.shap_value.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Grad-CAM tab */}
      {activeTab === 'gradcam' && (
        <div>
          {!gradcamData ? (
            <div style={{ padding: 24, textAlign: 'center', color: '#64748b' }}>Running Grad-CAM on 1D CNN...</div>
          ) : !gradcamData.available ? (
            <div style={{ padding: 20, color: '#92400e' }}>{gradcamData.note}</div>
          ) : (
            <>
              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: '#1e293b' }}>
                  🔥 Grad-CAM — Channel Importance Heatmap
                </h3>
                <p style={{ margin: '0 0 12px', fontSize: 12, color: '#64748b' }}>{gradcamData.description}</p>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={gradcamData.channel_importance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="grad_cam_score" fill="#f44336" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                {gradcamData.per_class_heatmaps && Object.entries(gradcamData.per_class_heatmaps).map(([cls, data]) => (
                  <div key={cls} style={cardStyle}>
                    <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>
                      {cls === 'Abnormal' ? '⚠️' : '✅'} {cls} — Channel Activation
                    </h3>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ fontSize: 12 }} />
                        <Bar dataKey="activation" fill={cls === 'Abnormal' ? '#f44336' : '#4caf50'} radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Feature Type Sensitivity (Input Gradients)</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={gradcamData.feature_type_sensitivity} layout="vertical" margin={{ left: 120 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="feature_type" tick={{ fontSize: 10 }} width={115} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="sensitivity" fill="#ff9800" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Sample Heatmaps (per-epoch Grad-CAM)</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
                  {gradcamData.sample_heatmaps?.map((s, i) => (
                    <div key={i} style={{ padding: 12, background: '#fef2f2', borderRadius: 8, border: '1px solid #fecaca' }}>
                      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
                        Epoch {s.sample_idx} — {s.predicted_class}
                        {s.predicted_class !== s.true_class && <span style={{ color: '#dc2626' }}> (true: {s.true_class})</span>}
                      </div>
                      {s.channel_activations?.map((c, j) => {
                        const maxAct = Math.max(...s.channel_activations.map(a => a.activation), 0.001)
                        const pct = (c.activation / maxAct) * 100
                        return (
                          <div key={j} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, padding: '2px 0' }}>
                            <span style={{ width: 60, color: '#334155', flexShrink: 0 }}>{c.channel}</span>
                            <div style={{ flex: 1, background: '#fee2e2', borderRadius: 4, height: 14 }}>
                              <div style={{ width: `${pct}%`, background: `rgba(244, 67, 54, ${0.3 + 0.7 * c.activation / maxAct})`,
                                borderRadius: 4, height: '100%' }} />
                            </div>
                            <span style={{ width: 45, textAlign: 'right', fontFamily: 'monospace', color: '#64748b' }}>
                              {c.activation.toFixed(3)}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ ...cardStyle, display: 'flex', gap: 20, fontSize: 12, color: '#64748b' }}>
                <div><strong>Model:</strong> {gradcamData.model?.type}</div>
                <div><strong>Accuracy:</strong> {(gradcamData.model?.accuracy * 100).toFixed(1)}%</div>
                <div><strong>Samples:</strong> {gradcamData.n_samples_explained}</div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Comparison tab */}
      {activeTab === 'comparison' && (
        <div>
          {!comparison ? (
            <div style={{ padding: 24, textAlign: 'center', color: '#64748b' }}>Computing multi-method comparison...</div>
          ) : !comparison.available ? (
            <div style={{ padding: 20, color: '#92400e' }}>{comparison.note}</div>
          ) : (
            <>
              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>⚖️ Consensus Ranking — Captum IG vs LIME vs SHAP</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: '#f1f5f9' }}>
                        <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Rank</th>
                        <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Feature</th>
                        <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#1e88e5' }}>Captum IG</th>
                        <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#4caf50' }}>LIME</th>
                        <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#ff9800' }}>SHAP</th>
                        <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', fontWeight: 700 }}>Consensus</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.comparison?.map((row, i) => (
                        <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                          <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{row.rank}</td>
                          <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontFamily: 'monospace', fontSize: 11 }}>{row.feature}</td>
                          <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', textAlign: 'right' }}>{row.captum_ig.toFixed(3)}</td>
                          <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', textAlign: 'right' }}>{row.lime.toFixed(3)}</td>
                          <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', textAlign: 'right' }}>{row.shap.toFixed(3)}</td>
                          <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', textAlign: 'right', fontWeight: 700 }}>{row.consensus.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Normalized Importance — Multi-Method Overlay</h3>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={comparison.comparison?.slice(0, 10)} margin={{ left: 140 }} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={135} />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="captum_ig" fill="#1e88e5" name="Captum IG" />
                    <Bar dataKey="lime" fill="#4caf50" name="LIME" />
                    <Bar dataKey="shap" fill="#ff9800" name="SHAP" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={cardStyle}>
                <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#1e293b' }}>Method Agreement (Kendall Tau)</h3>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  {comparison.agreement && Object.entries(comparison.agreement).filter(([k]) => k !== 'note').map(([k, v]) => (
                    <div key={k} style={{ padding: '12px 20px', background: '#f0f9ff', borderRadius: 8, border: '1px solid #bae6fd', textAlign: 'center' }}>
                      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: v > 0.5 ? '#16a34a' : v > 0.2 ? '#d97706' : '#dc2626' }}>
                        {v.toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>{comparison.agreement?.note}</p>
              </div>
            </>
          )}
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 20, padding: 12, background: '#f1f5f9', borderRadius: 8, fontSize: 11, color: '#64748b' }}>
        <strong>Libraries:</strong> Captum {overview.methods?.[0]?.version} (PyTorch) · LIME {overview.methods?.[1]?.version} · SHAP (TreeExplainer) ·
        Grad-CAM 1.5.5 (pytorch-grad-cam) · scikit-learn (GradientBoosting) · MNE (EEG I/O) · SciPy (Welch PSD)
        <br />
        <strong>Data:</strong> Real EEG features from {overview.data?.file} ({overview.data?.sfreq} Hz, {overview.data?.n_channels} channels, {overview.data?.n_epochs} epochs)
        <br />
        <strong>Compliance:</strong> EU AI Act Art. 86 — four independent explanation methods for triangulated attribution
      </div>
    </div>
  )
}
