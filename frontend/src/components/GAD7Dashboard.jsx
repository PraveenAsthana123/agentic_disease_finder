import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const SEV_COLORS = {
  minimal: '#22c55e', mild: '#84cc16', moderate: '#eab308', severe: '#ef4444'
}
const COLORS = ['#22c55e', '#84cc16', '#eab308', '#ef4444', '#8b5cf6']

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

function SevBadge({ level }) {
  const lbl = (level || 'unknown').toLowerCase()
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: (SEV_COLORS[lbl] || '#94a3b8') + '22', color: SEV_COLORS[lbl] || '#64748b'
    }}>{(level || 'unknown').toUpperCase()}</span>
  )
}

export default function GAD7Dashboard() {
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
          axios.get(`${API_URL}/gad7-dashboard/overview`),
          axios.get(`${API_URL}/gad7-dashboard/breakdown`),
          axios.get(`${API_URL}/gad7-dashboard/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading GAD-7 data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'items', label: 'Item Analysis' },
    { id: 'trends', label: 'Trends & History' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const sevData = overview?.severity_distribution
    ? Object.entries(overview.severity_distribution).map(([k, v]) => ({ name: k, value: v, color: SEV_COLORS[k] || '#94a3b8' }))
    : []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>GAD-7 Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Generalized Anxiety Disorder-7 — anxiety screening & severity monitoring
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
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
              <KPI label="Avg Score" value={fmt(overview.avg_score)} sub="range 0-21" />
              <KPI label="Moderate+ Rate" value={`${fmt(overview.pct_moderate_plus)}%`}
                color={overview.pct_moderate_plus > 40 ? '#ef4444' : overview.pct_moderate_plus > 20 ? '#f97316' : '#22c55e'} />
            </div>
          </Card>

          {/* Severity Distribution Pie */}
          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {sevData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {sevData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Active Alerts */}
          <Card title="Active Alerts (score >= 10)" span={2}>
            {overview.active_alerts?.length > 0 ? (
              <div style={{ maxHeight: 200, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Alert</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Score</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Severity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overview.active_alerts.map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '4px 8px', fontWeight: 600 }}>{a.patient_id}</td>
                        <td style={{ padding: '4px 8px', color: '#ef4444', fontSize: 11 }}>{a.alert}</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}>{a.score}</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}><SevBadge level={a.severity} /></td>
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
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Severity</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Interpretation</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.latest_score}/{p.max_score}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><SevBadge level={p.severity} /></td>
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

      {tab === 'items' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Item Endorsement Rates */}
          <Card title="Item Endorsement Rates (% scoring > 0)" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.item_endorsement} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="label" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="any_pct" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Any (1+)" />
                <Bar dataKey="frequent_pct" fill="#ef4444" radius={[0, 4, 4, 0]} name="Frequent (2+)" />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
              Blue = any endorsement (1-3), Red = frequent/nearly every day (2-3)
            </div>
          </Card>

          {/* Severity Transitions */}
          <Card title="Severity Transitions (patients with 2+ assessments)" span={2}>
            {breakdown.severity_transitions?.length > 0 ? (
              <div style={{ maxHeight: 280, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>First</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}></th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Latest</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Change</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>#</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.severity_transitions.map((t, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '4px 8px', fontWeight: 600 }}>{t.patient_id}</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                          {t.first_score} <SevBadge level={t.first_severity} />
                        </td>
                        <td style={{ padding: '4px 8px', textAlign: 'center', color: '#94a3b8' }}>&rarr;</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                          {t.latest_score} <SevBadge level={t.latest_severity} />
                        </td>
                        <td style={{ padding: '4px 8px', textAlign: 'center',
                          color: t.change < 0 ? '#22c55e' : t.change > 0 ? '#ef4444' : '#64748b',
                          fontWeight: 600 }}>
                          {t.change > 0 ? '+' : ''}{t.change}
                        </td>
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
          <Card title="Monthly Average Score & Moderate+ Rate">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={breakdown.trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="score" domain={[0, 21]} />
                <YAxis yAxisId="pct" orientation="right" domain={[0, 100]} unit="%" />
                <Tooltip />
                <Line yAxisId="score" type="monotone" dataKey="avg_score" stroke="#3b82f6" strokeWidth={2} name="Avg Score" dot />
                <Line yAxisId="pct" type="monotone" dataKey="pct_moderate_plus" stroke="#f97316" strokeWidth={2} strokeDasharray="5 5" name="Moderate+ %" dot />
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
                        <div style={{ fontWeight: 600 }}>{h.score}/21 <SevBadge level={h.severity} /></div>
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
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={defs.title}>
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 12px' }}>{defs.reference}</p>

            <h4 style={{ fontSize: 14, color: '#334155', margin: '16px 0 8px' }}>Items</h4>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b', width: 40 }}>#</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b', width: 150 }}>Domain</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Question</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b', width: 200 }}>Scoring</th>
                </tr>
              </thead>
              <tbody>
                {(defs.items || []).map((it, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{i + 1}</td>
                    <td style={{ padding: '6px 8px', fontWeight: 600, color: '#334155' }}>{it.label}</td>
                    <td style={{ padding: '6px 8px', color: '#475569' }}>{it.description}</td>
                    <td style={{ padding: '6px 8px', fontSize: 11, color: '#94a3b8' }}>{it.scoring}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <h4 style={{ fontSize: 14, color: '#334155', margin: '20px 0 8px' }}>Severity Tiers</h4>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Range</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Severity</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Recommended Action</th>
                </tr>
              </thead>
              <tbody>
                {(defs.severity_tiers || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{t.range[0]}-{t.range[1]}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <span style={{ color: t.color, fontWeight: 600 }}>{t.label}</span>
                    </td>
                    <td style={{ padding: '6px 8px', color: '#475569' }}>{t.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <h4 style={{ fontSize: 14, color: '#334155', margin: '20px 0 8px' }}>Clinical Notes</h4>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <tbody>
                {(defs.clinical_notes || []).map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 180 }}>{d.term}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
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
