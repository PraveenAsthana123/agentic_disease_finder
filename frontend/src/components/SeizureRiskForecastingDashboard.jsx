import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const TIER_COLORS = { critical: '#991b1b', high: '#ef4444', moderate: '#f59e0b', low: '#10b981' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 16, fontWeight: 600 }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, color }) {
  return (
    <div style={{ background: '#f8fafc', borderRadius: 8, padding: '14px 18px', textAlign: 'center', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{label}</div>
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 12, fontWeight: 600,
      background: (color || '#64748b') + '22', color: color || '#64748b'
    }}>{text}</span>
  )
}

export default function SeizureRiskForecastingDashboard() {
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
          axios.get(`${API_URL}/api/seizure-risk-forecast/overview`),
          axios.get(`${API_URL}/api/seizure-risk-forecast/breakdown`),
          axios.get(`${API_URL}/api/seizure-risk-forecast/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load seizure-risk forecasting data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading Seizure Risk Forecasting data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Seizure Risk Forecasting data not available.'}
    </div>
  )

  const k = overview.kpis || {}
  const tierPie = overview.charts?.risk_tier_pie || []
  const horizonBar = overview.charts?.horizon_bar || []
  const riskHist = overview.charts?.risk_distribution || []
  const escalationBar = overview.charts?.escalation_actions_bar || []
  const thresholds = overview.thresholds || []
  const patients = breakdown?.per_patient_forecasts || []
  const escalLog = breakdown?.escalation_log || []
  const fGaps = breakdown?.forecasting_gaps || []
  const concepts = defs || []

  const tabs = ['overview', 'patients', 'escalation', 'gaps', 'definitions']
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', fontWeight: active ? 700 : 400,
    borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
    color: active ? '#1e293b' : '#64748b', background: 'none', border: 'none',
    borderBottomWidth: 2, borderBottomStyle: 'solid',
    borderBottomColor: active ? '#3b82f6' : 'transparent'
  })

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 16, fontSize: 22, fontWeight: 700 }}>Seizure Risk Forecasting</h2>
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #e2e8f0', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={tabStyle(tab === t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 20 }}>
            <KPI label="Patients Monitored" value={k.total_patients_monitored} />
            <KPI label="Avg Risk Score" value={k.avg_risk_score} />
            <KPI label="Critical %" value={k.critical_pct + '%'} color="#991b1b" />
            <KPI label="High %" value={k.high_pct + '%'} color="#ef4444" />
            <KPI label="Patients Critical" value={k.patients_critical} color="#991b1b" />
            <KPI label="Patients High" value={k.patients_high} color="#ef4444" />
            <KPI label="Forecasting Gaps" value={k.forecasting_gaps_total} />
            <KPI label="Gaps Partial" value={k.gaps_partial} color="#f59e0b" />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
            <Card title="Risk Tier Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={tierPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {tierPie.map((e, i) => (
                      <Cell key={i} fill={TIER_COLORS[e.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Forecast Horizon Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={horizonBar}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Risk Score Distribution (Histogram)">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={riskHist}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Escalation Actions by Tier">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={escalationBar} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {escalationBar.map((e, i) => (
                      <Cell key={i} fill={TIER_COLORS[e.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Alert Thresholds Configuration">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Tier</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Min Score</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Max Response (min)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Escalation</th>
                  </tr>
                </thead>
                <tbody>
                  {thresholds.map((th, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px' }}>
                        <Badge text={th.tier} color={TIER_COLORS[th.tier]} />
                      </td>
                      <td style={{ padding: '8px 12px' }}>{th.min_score}</td>
                      <td style={{ padding: '8px 12px' }}>{th.max_response_min ?? 'N/A'}</td>
                      <td style={{ padding: '8px 12px' }}>{th.escalation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'patients' && (
        <Card title={`Per-Patient Risk Forecasts (${patients.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Patient</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Risk Score</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Tier</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Horizon</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Confidence</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{p.risk_score}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge text={p.risk_tier} color={TIER_COLORS[p.risk_tier]} />
                    </td>
                    <td style={{ padding: '8px 12px' }}>{p.forecast_horizon}</td>
                    <td style={{ padding: '8px 12px' }}>{p.model_confidence}</td>
                    <td style={{ padding: '8px 12px' }}>
                      {(p.escalation_actions || []).join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'escalation' && (
        <Card title={`Escalation Log (${escalLog.length} active)`}>
          {escalLog.length === 0 ? (
            <p style={{ color: '#64748b' }}>No escalations active.</p>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Patient</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Tier</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Score</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Horizon</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Response Target</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Actions Triggered</th>
                  </tr>
                </thead>
                <tbody>
                  {escalLog.map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{e.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <Badge text={e.risk_tier} color={TIER_COLORS[e.risk_tier]} />
                      </td>
                      <td style={{ padding: '8px 12px' }}>{e.risk_score}</td>
                      <td style={{ padding: '8px 12px' }}>{e.horizon}</td>
                      <td style={{ padding: '8px 12px' }}>{e.response_target_min} min</td>
                      <td style={{ padding: '8px 12px' }}>
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                          {(e.actions_triggered || []).map((a, j) => <li key={j}>{a}</li>)}
                        </ul>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}

      {tab === 'gaps' && (
        <Card title={`Forecasting Feature Gaps (${fGaps.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Feature</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Category</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Status</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Priority</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Why</th>
                </tr>
              </thead>
              <tbody>
                {fGaps.map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{g.feature}</td>
                    <td style={{ padding: '8px 12px' }}>{g.category}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge
                        text={g.in_project === true ? 'built' : g.in_project === false ? 'missing' : String(g.in_project)}
                        color={g.in_project === 'built' ? '#10b981' : g.in_project === 'partial' ? '#f59e0b' : '#ef4444'}
                      />
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge text={g.priority} color={g.priority === 'high' ? '#ef4444' : g.priority === 'medium' ? '#f59e0b' : '#3b82f6'} />
                    </td>
                    <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{g.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'definitions' && (
        <Card title="Definitions">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', width: '20%' }}>Term</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {concepts.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{c.term}</td>
                    <td style={{ padding: '8px 12px' }}>{c.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
