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
const STATUS_COLORS = { Completed: '#10b981', Partial: '#f59e0b', Failed: '#ef4444', Insufficient: '#94a3b8' }
const RISK_COLORS = ['#ef4444', '#f59e0b', '#10b981']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'by_patient', label: 'By Patient' },
  { id: 'sessions', label: 'Recent Sessions' },
  { id: 'methods', label: 'Fusion Methods' },
  { id: 'definitions', label: 'Definitions' },
]

export default function MultimodalFusionDashboard() {
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
      axios.get(`${API_URL}/api/multimodal-fusion/overview`),
      axios.get(`${API_URL}/api/multimodal-fusion/breakdown`),
      axios.get(`${API_URL}/api/multimodal-fusion/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Multimodal Fusion Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Multimodal Fusion Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        EEG + Video + Clinical Notes + Medication + MRI — fusion analysis, risk scoring, subtype prediction
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
            <KPI label="Total Sessions" value={fmt(ov.total_sessions)} sub={`${ov.total_patients} patients`} color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Completion Rate" value={pct(ov.completion_rate)} sub="Fully fused" color="#10b981" />
          </Card>
          <Card>
            <KPI label="Avg Risk Score" value={fmt(ov.avg_risk_score)} sub="0 = low, 1 = high" color="#ef4444" />
          </Card>
          <Card>
            <KPI label="Avg Confidence" value={fmt(ov.avg_confidence)} sub="Model certainty" color="#8b5cf6" />
          </Card>

          <Card>
            <KPI label="Avg Concordance" value={fmt(ov.avg_concordance)} sub="Cross-modal agreement" color="#06b6d4" />
          </Card>
          <Card>
            <KPI label="Avg Processing" value={`${fmt(ov.avg_processing_time_sec)}s`} sub="Per session" color="#f59e0b" />
          </Card>
          <Card>
            <KPI label="Avg Modalities" value={fmt(ov.avg_modalities_per_session)} sub="Out of 5" color="#334155" />
          </Card>
          <Card>
            <KPI label="Sessions/Patient" value={ov.total_patients ? (ov.total_sessions / ov.total_patients).toFixed(1) : '--'} sub="Average" color="#334155" />
          </Card>

          {/* Status Pie */}
          <Card title="Session Status" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={(ov.status_distribution || []).filter(d => d.count > 0)} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.status_distribution || []).filter(d => d.count > 0).map((d, i) => (
                    <Cell key={i} fill={STATUS_COLORS[d.status] || COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Risk Distribution */}
          <Card title="Risk Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.risk_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Sessions">
                  {(ov.risk_distribution || []).map((_, i) => <Cell key={i} fill={RISK_COLORS[i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Modality Availability */}
          <Card title="Modality Availability (%)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.modality_availability || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
                <YAxis dataKey="modality" type="category" width={90} tick={{ fontSize: 12 }} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="rate" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Fusion Methods */}
          <Card title="Fusion Methods Used" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.fusion_methods || []} dataKey="count" nameKey="method" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name.replace(/_/g, ' ')} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.fusion_methods || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Predicted Subtypes */}
          <Card title="Predicted Subtypes" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.predicted_subtypes || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subtype" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Monthly Trend */}
          <Card title="Monthly Session Volume" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={ov.monthly_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="sessions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── BY PATIENT TAB ─── */}
      {tab === 'by_patient' && bd.available && (
        <Card title="Per-Patient Fusion Summary" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Patient</th>
                  <th style={{ padding: '8px 12px' }}>Sessions</th>
                  <th style={{ padding: '8px 12px' }}>Completed</th>
                  <th style={{ padding: '8px 12px' }}>Rate</th>
                  <th style={{ padding: '8px 12px' }}>Avg Risk</th>
                  <th style={{ padding: '8px 12px' }}>Avg Confidence</th>
                  <th style={{ padding: '8px 12px' }}>Avg Modalities</th>
                </tr>
              </thead>
              <tbody>
                {(bd.per_patient || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{p.total_sessions}</td>
                    <td style={{ padding: '8px 12px' }}>{p.completed}</td>
                    <td style={{ padding: '8px 12px', color: p.completion_rate >= 80 ? '#10b981' : p.completion_rate >= 50 ? '#f59e0b' : '#ef4444' }}>{p.completion_rate}%</td>
                    <td style={{ padding: '8px 12px', color: p.avg_risk >= 0.7 ? '#ef4444' : p.avg_risk >= 0.3 ? '#f59e0b' : '#10b981' }}>{p.avg_risk}</td>
                    <td style={{ padding: '8px 12px' }}>{p.avg_confidence}</td>
                    <td style={{ padding: '8px 12px' }}>{p.avg_modalities}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── RECENT SESSIONS TAB ─── */}
      {tab === 'sessions' && bd.available && (
        <Card title="Recent Fusion Sessions (last 20)" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 8px' }}>Session</th>
                  <th style={{ padding: '6px 8px' }}>Patient</th>
                  <th style={{ padding: '6px 8px' }}>Date</th>
                  <th style={{ padding: '6px 8px' }}>Modalities</th>
                  <th style={{ padding: '6px 8px' }}>Method</th>
                  <th style={{ padding: '6px 8px' }}>Risk</th>
                  <th style={{ padding: '6px 8px' }}>Subtype</th>
                  <th style={{ padding: '6px 8px' }}>Confidence</th>
                  <th style={{ padding: '6px 8px' }}>Concordance</th>
                  <th style={{ padding: '6px 8px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(bd.recent_sessions || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>{s.session_id}</td>
                    <td style={{ padding: '6px 8px' }}>{s.patient_id}</td>
                    <td style={{ padding: '6px 8px' }}>{s.date}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'center' }}>{s.modalities}/5</td>
                    <td style={{ padding: '6px 8px' }}>{s.fusion_method.replace(/_/g, ' ')}</td>
                    <td style={{ padding: '6px 8px', color: s.risk_score >= 0.7 ? '#ef4444' : s.risk_score >= 0.3 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>{s.risk_score ?? '--'}</td>
                    <td style={{ padding: '6px 8px' }}>{s.subtype}</td>
                    <td style={{ padding: '6px 8px' }}>{s.confidence ?? '--'}</td>
                    <td style={{ padding: '6px 8px' }}>{s.concordance ?? '--'}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                        background: s.status === 'completed' ? '#dcfce7' : s.status === 'partial' ? '#fef3c7' : s.status === 'failed' ? '#fee2e2' : '#f1f5f9',
                        color: s.status === 'completed' ? '#166534' : s.status === 'partial' ? '#92400e' : s.status === 'failed' ? '#991b1b' : '#64748b',
                      }}>{s.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── FUSION METHODS TAB ─── */}
      {tab === 'methods' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Method Comparison" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Method</th>
                    <th style={{ padding: '8px 12px' }}>Sessions</th>
                    <th style={{ padding: '8px 12px' }}>Avg Risk</th>
                    <th style={{ padding: '8px 12px' }}>Avg Confidence</th>
                    <th style={{ padding: '8px 12px' }}>Avg Concordance</th>
                    <th style={{ padding: '8px 12px' }}>Avg Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.method_comparison || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{m.method.replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px' }}>{m.sessions}</td>
                      <td style={{ padding: '8px 12px' }}>{m.avg_risk}</td>
                      <td style={{ padding: '8px 12px' }}>{m.avg_confidence}</td>
                      <td style={{ padding: '8px 12px' }}>{m.avg_concordance}</td>
                      <td style={{ padding: '8px 12px' }}>{m.avg_processing_sec}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Method Performance (Confidence)" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.method_comparison || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="method" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="avg_confidence" fill="#8b5cf6" name="Avg Confidence" />
                <Bar dataKey="avg_concordance" fill="#06b6d4" name="Avg Concordance" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Modality Co-occurrence" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'center' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '6px 8px' }}></th>
                    {['EEG', 'Video', 'Notes', 'Meds', 'MRI'].map(h => (
                      <th key={h} style={{ padding: '6px 8px' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.modality_cooccurrence || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, textAlign: 'left' }}>{row.modality}</td>
                      {['EEG', 'Video', 'Notes', 'Meds', 'MRI'].map(h => (
                        <td key={h} style={{ padding: '6px 8px', background: `rgba(59,130,246,${Math.min(row[h] / (ov.total_sessions || 1), 1) * 0.3})` }}>{row[h]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Glossary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px', width: 200 }}>Term</th>
                  <th style={{ padding: '8px 12px' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{g.term}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Risk Tiers">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Tier</th>
                  <th style={{ padding: '8px 12px' }}>Range</th>
                  <th style={{ padding: '8px 12px' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {(defs.risk_tiers || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: i === 0 ? '#ef4444' : i === 1 ? '#f59e0b' : '#10b981' }}>{t.tier}</td>
                    <td style={{ padding: '8px 12px' }}>{t.range}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{t.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Data Modalities">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Modality</th>
                  <th style={{ padding: '8px 12px' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.modalities || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{m.description}</td>
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
