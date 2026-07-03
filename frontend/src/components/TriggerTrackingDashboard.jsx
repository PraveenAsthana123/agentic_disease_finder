import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')

const RISK_COLORS = { low: '#2ecc71', moderate: '#f39c12', high: '#e67e22', critical: '#e74c3c' }
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

function RiskBadge({ level }) {
  const key = level?.toLowerCase()
  return <Badge text={level || '--'} color={RISK_COLORS[key] || '#64748b'} />
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'trigger_analysis', label: 'Trigger Analysis' },
  { id: 'patient_risk', label: 'Patient Risk' },
  { id: 'patient_detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function TriggerTrackingDashboard() {
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
      axios.get(`${API_URL}/api/trigger-tracking/overview`),
      axios.get(`${API_URL}/api/trigger-tracking/breakdown`),
      axios.get(`${API_URL}/api/trigger-tracking/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Trigger Tracking Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Trigger Tracking Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Seizure trigger monitoring — sleep, stress, medication adherence, risk stratification, and temporal trends
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI Row 1 */}
          <Card title="Total Logs"><KPI value={ov.total_logs} label="Log entries" color="#3b82f6" /></Card>
          <Card title="Patients Tracked"><KPI value={ov.patients_tracked} label="Unique patients" color="#8b5cf6" /></Card>
          <Card title="Avg Seizure Rate"><KPI value={ov.avg_seizure_rate != null ? `${ov.avg_seizure_rate.toFixed(1)}%` : '--'} label="Seizure rate" color="#e74c3c" /></Card>
          <Card title="Avg Sleep"><KPI value={ov.avg_sleep != null ? `${ov.avg_sleep.toFixed(1)} hrs` : '--'} label="Hours per night" color="#10b981" /></Card>

          {/* KPI Row 2 */}
          <Card title="Avg Stress Level"><KPI value={ov.avg_stress != null ? ov.avg_stress.toFixed(1) : '--'} label="Stress (1-10)" color="#f59e0b" /></Card>
          <Card title="Med Adherence"><KPI value={ov.med_adherence != null ? `${ov.med_adherence.toFixed(1)}%` : '--'} label="Medication adherence" color="#06b6d4" /></Card>
          <Card title="Top Trigger"><KPI value={ov.top_trigger || '--'} label="Most common trigger" color="#ec4899" /></Card>
          <Card title="Days Tracked"><KPI value={ov.days_tracked} label="Tracking period" color="#f97316" /></Card>

          {/* Trigger Distribution Bar Chart */}
          <Card title="Trigger Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.trigger_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="trigger" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count">
                  {(ov.trigger_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Risk Level Distribution Pie Chart */}
          <Card title="Risk Level Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.risk_distribution || []} dataKey="count" nameKey="level" cx="50%" cy="50%" outerRadius={80} label={e => `${e.level}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.risk_distribution || []).map((e, i) => (
                    <Cell key={i} fill={RISK_COLORS[e.level?.toLowerCase()] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Factor Correlation Bar Chart */}
          <Card title="Factor Correlation" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.factor_correlation || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="factor" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="correlation" name="Correlation" radius={[4, 4, 0, 0]}>
                  {(ov.factor_correlation || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Adherence Impact Bar Chart */}
          <Card title="Adherence Impact" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.adherence_impact || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="group" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="seizure_rate" name="Seizure Rate %" radius={[4, 4, 0, 0]}>
                  {(ov.adherence_impact || []).map((e, i) => (
                    <Cell key={i} fill={i === 0 ? '#2ecc71' : '#e74c3c'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* TRIGGER ANALYSIS TAB */}
      {tab === 'trigger_analysis' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Sleep vs Seizure Rate Bar Chart */}
          <Card title="Sleep vs Seizure Rate">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.sleep_vs_seizure || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="sleep_hours" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="seizure_pct" name="Seizure %" radius={[4, 4, 0, 0]}>
                  {(bd.sleep_vs_seizure || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Temporal Trend Line Chart */}
          <Card title="Temporal Trend (Daily Seizure Count — 90 Days)">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={bd.temporal_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={10} angle={-30} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Line type="monotone" dataKey="seizure_count" name="Seizures" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Factor Correlation Table */}
          <Card title="Factor Correlation Coefficients">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Factor', 'Correlation Coefficient'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.factor_correlation || []).map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(f.factor)}</td>
                      <td style={{ padding: '6px', fontWeight: 600, color: f.correlation > 0 ? '#e74c3c' : '#2ecc71' }}>
                        {f.correlation != null ? f.correlation.toFixed(3) : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* PATIENT RISK TAB */}
      {tab === 'patient_risk' && (
        <Card title="Patient Risk Overview">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient ID', 'Total Logs', 'Seizures', 'Seizure Rate', 'Avg Sleep', 'Avg Stress', 'Adherence %', 'Risk Level', 'Top Triggers'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(p.patient_id)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.total_logs)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.seizures)}</td>
                    <td style={{ padding: '6px', fontWeight: 600, color: '#e74c3c' }}>
                      {p.seizure_rate != null ? `${p.seizure_rate.toFixed(1)}%` : '--'}
                    </td>
                    <td style={{ padding: '6px' }}>{p.avg_sleep != null ? p.avg_sleep.toFixed(1) : '--'}</td>
                    <td style={{ padding: '6px' }}>{p.avg_stress != null ? p.avg_stress.toFixed(1) : '--'}</td>
                    <td style={{ padding: '6px' }}>{p.adherence != null ? `${p.adherence.toFixed(1)}%` : '--'}</td>
                    <td style={{ padding: '6px' }}><RiskBadge level={p.risk_level} /></td>
                    <td style={{ padding: '6px', maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {Array.isArray(p.top_triggers) ? p.top_triggers.join(', ') : fmt(p.top_triggers)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patient_detail' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {(bd.patients || []).slice(0, 5).map((p, pi) => (
            <Card key={pi} title={`Patient ${p.patient_id}`}>
              {/* Stats Summary */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12, marginBottom: 16 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#3b82f6' }}>{fmt(p.total_logs)}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Total Logs</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#e74c3c' }}>{fmt(p.seizures)}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Seizures</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#e74c3c' }}>{p.seizure_rate != null ? `${p.seizure_rate.toFixed(1)}%` : '--'}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Seizure Rate</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#10b981' }}>{p.avg_sleep != null ? p.avg_sleep.toFixed(1) : '--'}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Avg Sleep (hrs)</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#f59e0b' }}>{p.avg_stress != null ? p.avg_stress.toFixed(1) : '--'}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Avg Stress</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700 }}><RiskBadge level={p.risk_level} /></div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Risk Level</div>
                </div>
              </div>

              {/* Recent Logs Table */}
              <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Recent Logs (Last 7 Entries)</h4>
              {(p.recent_logs || []).length > 0 ? (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: '#f1f5f9' }}>
                        {['Date', 'Sleep (hrs)', 'Stress', 'Medication', 'Seizure', 'Triggers'].map(h => (
                          <th key={h} style={{ padding: '6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(p.recent_logs || []).slice(0, 7).map((log, li) => (
                        <tr key={li} style={{ borderBottom: '1px solid #e2e8f0' }}>
                          <td style={{ padding: '4px 6px' }}>{fmt(log.date)}</td>
                          <td style={{ padding: '4px 6px' }}>{fmt(log.sleep)}</td>
                          <td style={{ padding: '4px 6px' }}>{fmt(log.stress)}</td>
                          <td style={{ padding: '4px 6px' }}>
                            <Badge
                              text={log.medication ? 'Yes' : 'No'}
                              color={log.medication ? '#2ecc71' : '#e74c3c'}
                            />
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            <Badge
                              text={log.seizure ? 'Yes' : 'No'}
                              color={log.seizure ? '#e74c3c' : '#2ecc71'}
                            />
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {Array.isArray(log.triggers) ? log.triggers.join(', ') : fmt(log.triggers)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: '#94a3b8', fontSize: 12 }}>No recent logs available for this patient.</p>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Trigger Tracking Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
