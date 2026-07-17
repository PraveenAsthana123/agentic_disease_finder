import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function FallRiskBadge({ level }) {
  const colors = { high: '#ef4444', moderate: '#f59e0b', low: '#10b981' }
  const color = colors[level] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{level || '--'}</span>
  )
}

function AgreeBadge({ agree }) {
  const color = agree ? '#10b981' : '#ef4444'
  const label = agree ? 'AGREE' : 'DISAGREE'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{label}</span>
  )
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export default function SeizureSemiologyDashboard() {
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
          axios.get(`${API_URL}/seizure-semiology/overview`),
          axios.get(`${API_URL}/seizure-semiology/breakdown`),
          axios.get(`${API_URL}/seizure-semiology/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load seizure semiology data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading seizure semiology data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Seizure semiology data not available</div>

  const tabs = ['overview', 'patients', 'classification', 'definitions']
  const lat = overview.lateralisation || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Seizure Semiology Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        AI-driven seizure-type classification from video-EEG motor patterns — {fmt(overview.total_events_classified)} events across {fmt(overview.total_patients)} patients
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Total Events Classified" value={overview.total_events_classified} />
              <KPI label="Semiology Types Detected" value={overview.semiology_types_detected} color="#8b5cf6" />
              <KPI label="Avg Confidence" value={overview.average_confidence} sub="score" color="#3b82f6" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Fall Risk Events" value={overview.fall_risk_events} color="#ef4444" />
              <KPI label="Fall Risk %" value={overview.fall_risk_pct} sub="%" color="#f59e0b" />
              <KPI label="Total Patients" value={overview.total_patients} color="#10b981" />
            </div>
          </Card>

          <Card title="Semiology Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview.type_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="type" type="category" tick={{ fontSize: 11 }} width={130} />
                <Tooltip />
                <Bar dataKey="count" name="Events" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {(overview.type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Epileptogenic Zone Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview.zone_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="zone" type="category" tick={{ fontSize: 10 }} width={180} />
                <Tooltip />
                <Bar dataKey="count" name="Events" fill="#8b5cf6" radius={[0, 4, 4, 0]}>
                  {(overview.zone_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Lateralisation" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Left', value: lat.left || 0 },
                    { name: 'Right', value: lat.right || 0 },
                    { name: 'Bilateral', value: lat.bilateral || 0 }
                  ]}
                  cx="50%" cy="50%" outerRadius={90} dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  <Cell fill="#3b82f6" />
                  <Cell fill="#ef4444" />
                  <Cell fill="#f59e0b" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Confidence Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.confidence_histogram || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Events" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Model Performance Comparison" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Model</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Type</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Accuracy</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Macro F1</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>AUC-ROC</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Inference (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.model_performance || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{m.model}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.type}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: m.accuracy >= 0.85 ? '#10b981' : '#f59e0b' }}>{(m.accuracy * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{(m.macro_f1 * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{(m.auc_roc * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#64748b' }}>{m.inference_ms}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Per-Class Metrics (Best Model)" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Semiology Type</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Precision</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Recall</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>F1</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.per_class_metrics || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{c.type}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{(c.precision * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{(c.recall * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: c.f1 >= 0.8 ? '#10b981' : c.f1 >= 0.65 ? '#f59e0b' : '#ef4444' }}>{(c.f1 * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#64748b' }}>{c.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Patients tab ── */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Patient Semiology Profiles" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Age</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Sex</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Events</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Conf</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Fall Risk</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>AI-Clinician Agree</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Types Detected</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_profiles || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.name || p.patient_id}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{p.age || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.sex || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{p.total_events}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: p.avg_confidence >= 0.8 ? '#10b981' : '#f59e0b' }}>{(p.avg_confidence * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><FallRiskBadge level={p.fall_risk_level} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(p.ai_clinician_agreement_pct)}%</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{(p.types_detected || []).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="High Fall Risk Patients" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#fef2f2' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Patient</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Fall Risk Score</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#991b1b' }}>Risk Level</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Events</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Types</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_profiles || []).filter(p => p.fall_risk_level === 'high' || p.fall_risk_level === 'moderate').sort((a, b) => b.cumulative_fall_risk - a.cumulative_fall_risk).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.name || p.patient_id}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>{p.cumulative_fall_risk}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><FallRiskBadge level={p.fall_risk_level} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{p.total_events}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{(p.types_detected || []).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Per-Patient Event Detail" span={4}>
            {(breakdown.patient_profiles || []).slice(0, 5).map((p, pi) => (
              <div key={pi} style={{ marginBottom: 16 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 14, color: '#1e293b' }}>
                  {p.name || p.patient_id} — {p.total_events} events, Fall risk: <FallRiskBadge level={p.fall_risk_level} />
                </h4>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                        <th style={{ textAlign: 'left', padding: '6px 10px', color: '#475569' }}>Event</th>
                        <th style={{ textAlign: 'left', padding: '6px 10px', color: '#475569' }}>Semiology Type</th>
                        <th style={{ textAlign: 'right', padding: '6px 10px', color: '#475569' }}>Confidence</th>
                        <th style={{ textAlign: 'center', padding: '6px 10px', color: '#475569' }}>Lateralisation</th>
                        <th style={{ textAlign: 'left', padding: '6px 10px', color: '#475569' }}>Inferred Zone</th>
                        <th style={{ textAlign: 'center', padding: '6px 10px', color: '#475569' }}>AI vs Clinician</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(p.events || []).map((e, ei) => (
                        <tr key={ei} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '6px 10px' }}>#{e.event_id}</td>
                          <td style={{ padding: '6px 10px', fontWeight: 500 }}>{e.semiology_type}</td>
                          <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600, color: e.confidence >= 0.8 ? '#10b981' : '#f59e0b' }}>{(e.confidence * 100).toFixed(0)}%</td>
                          <td style={{ padding: '6px 10px', textAlign: 'center' }}>{e.lateralisation}</td>
                          <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{e.inferred_zone}</td>
                          <td style={{ padding: '6px 10px', textAlign: 'center' }}><AgreeBadge agree={e.ai_clinician_agree} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </Card>
        </div>
      )}

      {/* ── Classification tab ── */}
      {tab === 'classification' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Confusion Matrix (Best Model)" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569', fontWeight: 700 }}>True \ Predicted</th>
                    {(breakdown.confusion_labels || []).map((l, i) => (
                      <th key={i} style={{ textAlign: 'center', padding: '6px 4px', color: '#475569', fontSize: 10, maxWidth: 70, overflow: 'hidden' }}>{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.confusion_labels || []).map((trueLabel, ri) => {
                    const row = (breakdown.confusion_matrix || {})[trueLabel] || {}
                    return (
                      <tr key={ri} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 10 }}>{trueLabel}</td>
                        {(breakdown.confusion_labels || []).map((predLabel, ci) => {
                          const val = row[predLabel] || 0
                          const isDiag = ri === ci
                          return (
                            <td key={ci} style={{
                              textAlign: 'center', padding: '6px 4px',
                              fontWeight: isDiag ? 700 : 400,
                              color: isDiag ? '#10b981' : val > 3 ? '#ef4444' : '#94a3b8',
                              background: isDiag ? '#f0fdf4' : val > 3 ? '#fef2f2' : undefined
                            }}>{val}</td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Per-Class F1 Scores" span={4}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview.per_class_metrics || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 10, angle: -30 }} height={60} />
                <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Bar dataKey="precision" name="Precision" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="recall" name="Recall" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Semiology Types" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {(defs.semiology_types || []).map((s, i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{s.type}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>{s.description}</div>
                  <div style={{ fontSize: 11, color: '#475569' }}>
                    <strong>Zone:</strong> {s.localisation_zone} | <strong>Lateralising:</strong> {s.lateralising ? 'Yes' : 'No'} | <strong>Fall risk:</strong> {s.fall_risk_weight}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Classification Pipeline" span={1}>
            <ol style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#334155' }}>
              {(defs.classification_methodology?.pipeline_steps || []).map((step, i) => (
                <li key={i} style={{ marginBottom: 6 }}>{step.replace(/^\d+\.\s*/, '')}</li>
              ))}
            </ol>
          </Card>

          <Card title="Model Architectures" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Model</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.classification_methodology?.models || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 500 }}>{m.name}</td>
                      <td style={{ padding: '6px 8px' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 8,
                          background: m.status === 'built' ? '#dcfce7' : '#fef3c7',
                          color: m.status === 'built' ? '#166534' : '#92400e',
                          fontSize: 11, fontWeight: 600
                        }}>{m.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Fall Risk Scoring" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Level</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#475569' }}>Score Range</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Recommended Action</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.fall_risk_scoring?.levels || []).map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{l.level}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: '#64748b' }}>{l.range}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11 }}>{l.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p style={{ margin: '8px 0 0', fontSize: 11, color: '#94a3b8' }}>{defs.fall_risk_scoring?.note}</p>
          </Card>

          <Card title="ILAE Classification Mapping" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>ILAE Category</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Semiology Types</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.ilae_classification_mapping || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 500 }}>{m.ilae_category}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{(m.semiology_types || []).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="References" span={2}>
            <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </div>
      )}
    </div>
  )
}
