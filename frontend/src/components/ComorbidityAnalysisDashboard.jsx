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
  { id: 'risk_analysis', label: 'Risk Analysis' },
  { id: 'patient_profiles', label: 'Patient Profiles' },
  { id: 'co_occurrence', label: 'Co-occurrence' },
  { id: 'definitions', label: 'Definitions' },
]

const SEVERITY_COLORS = {
  minimal: { bg: '#dcfce7', text: '#166534' },
  mild: { bg: '#e0f2fe', text: '#075985' },
  moderate: { bg: '#fef3c7', text: '#92400e' },
  severe: { bg: '#fee2e2', text: '#991b1b' },
}

const HEATMAP_COLOR = (value, max) => {
  if (value == null || max === 0) return '#f1f5f9'
  const ratio = Math.min(value / max, 1)
  if (ratio < 0.25) return '#dbeafe'
  if (ratio < 0.5) return '#93c5fd'
  if (ratio < 0.75) return '#3b82f6'
  return '#1e40af'
}

export default function ComorbidityAnalysisDashboard() {
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
      axios.get(`${API_URL}/api/comorbidity-analysis/overview`),
      axios.get(`${API_URL}/api/comorbidity-analysis/breakdown`),
      axios.get(`${API_URL}/api/comorbidity-analysis/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Comorbidity Analysis Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Comorbidity Analysis Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Multi-condition co-occurrence analysis — risk stratification, treatment overlap, functional impact assessment
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
            <KPI label="Patients Screened" value={fmt(ov?.kpis?.patients_screened)} sub="Total evaluated" color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Total Comorbidities" value={fmt(ov?.kpis?.total_comorbidities)} sub="Unique conditions found" color="#ef4444" />
          </Card>
          <Card>
            <KPI label="Avg Comorbidity Count" value={fmt(ov?.kpis?.avg_comorbidity_count)} sub="Per patient" color="#8b5cf6" />
          </Card>
          <Card>
            <KPI label="Avg Risk Score" value={fmt(ov?.kpis?.avg_risk_score)} sub="Behavioral risk index" color="#f59e0b" />
          </Card>

          {/* Top Comorbidities Bar Chart */}
          <Card title="Top Comorbidities by Frequency" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={ov?.top_comorbidities || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" name="Frequency" fill="#3b82f6">
                  {(ov?.top_comorbidities || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Risk Severity Distribution Pie */}
          <Card title="Risk Severity Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={(ov?.risk_severity_distribution || []).filter(d => d.value > 0)}
                  dataKey="value"
                  nameKey="severity"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.risk_severity_distribution || []).filter(d => d.value > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Treatment Status Pie */}
          <Card title="Treatment Status">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={(ov?.treatment_status_distribution || []).filter(d => d.value > 0)}
                  dataKey="value"
                  nameKey="status"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.treatment_status_distribution || []).filter(d => d.value > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── RISK ANALYSIS TAB ─── */}
      {tab === 'risk_analysis' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Risk Score Histogram */}
          <Card title="Risk Score Histogram" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={bd?.risk_score_histogram || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" fill="#3b82f6">
                  {(bd?.risk_score_histogram || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Avg Risk Score by Comorbidity Count */}
          <Card title="Avg Risk Score by Comorbidity Count" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={bd?.risk_by_comorbidity_count || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="comorbidity_count" tick={{ fontSize: 11 }} label={{ value: 'Comorbidity Count', position: 'insideBottomRight', offset: -5, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="avg_risk_score" name="Avg Risk Score" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} />
                <Bar dataKey="n" name="Patient Count" fill="#3b82f6" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Functional Impact Radar */}
          <Card title="Functional Impact Distribution" span={1}>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={bd?.functional_impact || []}>
                <PolarGrid />
                <PolarAngleAxis dataKey="impact" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis tick={{ fontSize: 10 }} />
                <Radar name="Patients" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.2} strokeWidth={2} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          {/* Screening Instrument Usage */}
          <Card title="Screening Instrument Usage" span={1}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={bd?.screening_instruments || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" name="Times Used" fill="#10b981">
                  {(bd?.screening_instruments || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── PATIENT PROFILES TAB ─── */}
      {tab === 'patient_profiles' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Per-Patient Comorbidity Profiles" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Patient ID</th>
                    <th style={{ padding: '8px 12px' }}>Comorbidity Count</th>
                    <th style={{ padding: '8px 12px' }}>Risk Severity</th>
                    <th style={{ padding: '8px 12px' }}>Behavioral Risk Score</th>
                    <th style={{ padding: '8px 12px' }}>Conditions</th>
                    <th style={{ padding: '8px 12px' }}>Treatment Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.per_patient || []).map((p, i) => {
                    const sev = (p.risk_severity || '').toLowerCase()
                    const sevStyle = SEVERITY_COLORS[sev] || { bg: '#f1f5f9', text: '#475569' }
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(p.comorbidity_count)}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <span style={{
                            padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                            background: sevStyle.bg, color: sevStyle.text,
                          }}>{p.risk_severity ?? '--'}</span>
                        </td>
                        <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(p.behavioral_risk_score)}</td>
                        <td style={{ padding: '8px 12px', fontSize: 12, color: '#475569', maxWidth: 260 }}>
                          {Array.isArray(p.conditions) ? p.conditions.join(', ') : (p.conditions ?? '--')}
                        </td>
                        <td style={{ padding: '8px 12px' }}>
                          <span style={{
                            padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                            background: p.treatment_status === 'stable' ? '#dcfce7' : p.treatment_status === 'untreated' ? '#fef3c7' : p.treatment_status === 'treatment_resistant' ? '#fee2e2' : '#f1f5f9',
                            color: p.treatment_status === 'stable' ? '#166534' : p.treatment_status === 'untreated' ? '#92400e' : p.treatment_status === 'treatment_resistant' ? '#991b1b' : '#475569',
                          }}>{(p.treatment_status || '--').replace(/_/g, ' ')}</span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── CO-OCCURRENCE TAB ─── */}
      {tab === 'co_occurrence' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Co-occurrence Heatmap */}
          <Card title="Comorbidity Co-occurrence Matrix" span={2}>
            {(() => {
              const matrix = bd?.co_occurrence_matrix || {}
              const labels = matrix.labels || []
              const data = matrix.data || []
              const maxVal = Math.max(...data.flat().filter(v => v != null), 1)
              if (labels.length === 0) return <p style={{ color: '#94a3b8' }}>No co-occurrence data available</p>
              return (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '6px 8px', minWidth: 100 }}></th>
                        {labels.map((l, i) => (
                          <th key={i} style={{ padding: '6px 8px', transform: 'rotate(-45deg)', transformOrigin: 'left bottom', whiteSpace: 'nowrap', fontSize: 10, color: '#334155' }}>{l}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {labels.map((rowLabel, ri) => (
                        <tr key={ri}>
                          <td style={{ padding: '6px 8px', fontWeight: 600, color: '#334155', whiteSpace: 'nowrap' }}>{rowLabel}</td>
                          {(data[ri] || []).map((val, ci) => (
                            <td key={ci} style={{
                              padding: '6px 10px', textAlign: 'center', fontWeight: 600, fontSize: 11,
                              background: HEATMAP_COLOR(val, maxVal),
                              color: val != null && val / maxVal > 0.5 ? '#fff' : '#334155',
                              borderRadius: 4, border: '1px solid #f1f5f9',
                            }}>{val ?? '-'}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )
            })()}
          </Card>

          {/* Demographics: By Age Group */}
          <Card title="Comorbidity Rate by Age Group">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={bd?.demographics_age || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age_group" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="comorbidity_rate" name="Comorbidity Rate %" fill="#3b82f6" />
                <Bar dataKey="avg_risk_score" name="Avg Risk Score" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Demographics: By Gender */}
          <Card title="Comorbidity Rate by Gender">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={bd?.demographics_gender || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gender" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="comorbidity_rate" name="Comorbidity Rate %" fill="#8b5cf6" />
                <Bar dataKey="avg_risk_score" name="Avg Risk Score" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Comorbidity Analysis Terminology" span={2}>
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
