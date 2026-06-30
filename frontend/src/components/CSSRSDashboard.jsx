import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const RISK_COLORS = { none: '#22c55e', low: '#eab308', moderate: '#f97316', high: '#ef4444' }
const COLORS = ['#22c55e', '#eab308', '#f97316', '#ef4444', '#8b5cf6', '#3b82f6']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function RiskBadge({ level }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: (RISK_COLORS[level] || '#94a3b8') + '22', color: RISK_COLORS[level] || '#64748b'
    }}>{(level || 'unknown').toUpperCase()}</span>
  )
}

export default function CSSRSDashboard() {
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
          axios.get(`${API_URL}/cssrs-dashboard/overview`),
          axios.get(`${API_URL}/cssrs-dashboard/breakdown`),
          axios.get(`${API_URL}/cssrs-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading C-SSRS data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'screening', label: 'Screening & Intensity' },
    { id: 'trends', label: 'Trends & History' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const riskData = overview?.risk_distribution
    ? Object.entries(overview.risk_distribution).map(([k, v]) => ({ name: k, value: v, color: RISK_COLORS[k] || '#94a3b8' }))
    : []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>C-SSRS Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Columbia Suicide Severity Rating Scale — suicidal ideation & behavior screening
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #ef4444' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#ef4444' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Assessments" value={fmt(overview.total_assessments)} />
              <KPI label="Unique Patients" value={fmt(overview.unique_patients)} />
              <KPI label="Avg Score" value={fmt(overview.avg_score)} sub="range 0-31" />
              <KPI label="Ideation Rate" value={`${fmt(overview.ideation_rate_pct)}%`}
                color={overview.ideation_rate_pct > 50 ? '#ef4444' : overview.ideation_rate_pct > 25 ? '#f97316' : '#22c55e'} />
            </div>
          </Card>

          {/* Risk Distribution Pie */}
          <Card title="Risk Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={riskData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {riskData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {riskData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Active Alerts */}
          <Card title="Active Alerts" span={2}>
            {overview.active_alerts?.length > 0 ? (
              <div style={{ maxHeight: 200, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Alert</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Score</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overview.active_alerts.map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '4px 8px', fontWeight: 600 }}>{a.patient_id}</td>
                        <td style={{ padding: '4px 8px', color: '#ef4444', fontSize: 11 }}>{a.alert}</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}>{a.score}</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}><RiskBadge level={a.level} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#22c55e', fontSize: 13 }}>No active alerts</div>
            )}
          </Card>

          {/* Per-Patient Summary */}
          <Card title="Per-Patient Latest Scores" span={3}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Score</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Risk</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Interpretation</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.latest_score}/{p.max_score}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><RiskBadge level={p.level} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{p.interpretation}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#94a3b8' }}>{(p.assessed_at || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'screening' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Screening Endorsement Rates */}
          <Card title="Screening Item Endorsement Rates" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.screening_rates} layout="vertical" margin={{ left: 160 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="rate_pct" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Intensity Profile */}
          <Card title="Intensity Profile (among those with ideation)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown.intensity_summary}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 5]} />
                <Tooltip />
                <Bar dataKey="avg" fill="#f97316" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
              Scale: 1 (least severe) to 5 (most severe)
            </div>
          </Card>

          {/* Risk Transitions */}
          <Card title="Risk Level Transitions (patients with 2+ assessments)">
            {breakdown.risk_transitions?.length > 0 ? (
              <div style={{ maxHeight: 220, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>First</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}></th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Latest</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>#</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.risk_transitions.map((t, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '4px 8px', fontWeight: 600 }}>{t.patient_id}</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}><RiskBadge level={t.first_level} /></td>
                        <td style={{ padding: '4px 8px', textAlign: 'center', color: '#94a3b8' }}>→</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}><RiskBadge level={t.latest_level} /></td>
                        <td style={{ padding: '4px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{t.assessments}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No patients with multiple assessments yet</div>
            )}
          </Card>
        </div>
      )}

      {tab === 'trends' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Monthly Trend */}
          <Card title="Monthly Average Score & Ideation Rate">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={breakdown.trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="score" domain={[0, 31]} />
                <YAxis yAxisId="pct" orientation="right" domain={[0, 100]} unit="%" />
                <Tooltip />
                <Line yAxisId="score" type="monotone" dataKey="avg_score" stroke="#ef4444" strokeWidth={2} name="Avg Score" dot />
                <Line yAxisId="pct" type="monotone" dataKey="ideation_pct" stroke="#f97316" strokeWidth={2} strokeDasharray="5 5" name="Ideation %" dot />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Patient History */}
          <Card title="Per-Patient Assessment History">
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              {Object.entries(breakdown.patient_history || {}).map(([pid, hist]) => (
                <div key={pid} style={{ marginBottom: 16, padding: '10px 12px', background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>{pid}</div>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    {hist.map((h, i) => (
                      <div key={i} style={{ fontSize: 11, padding: '4px 8px', background: '#fff', borderRadius: 6, border: '1px solid #e2e8f0' }}>
                        <div style={{ fontWeight: 600 }}>{h.score} <RiskBadge level={h.level} /></div>
                        <div style={{ color: '#94a3b8', marginTop: 2 }}>{(h.date || '').slice(0, 10)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <Card title={defs.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {(defs.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 180 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}
