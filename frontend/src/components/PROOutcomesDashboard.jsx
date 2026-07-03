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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'sleep_fatigue', label: 'Sleep & Fatigue' },
  { id: 'mood_anxiety', label: 'Mood & Anxiety' },
  { id: 'patient_detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PROOutcomesDashboard() {
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
      axios.get(`${API_URL}/api/pro-outcomes/overview`),
      axios.get(`${API_URL}/api/pro-outcomes/breakdown`),
      axios.get(`${API_URL}/api/pro-outcomes/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading PRO Outcomes Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Patient-Reported Outcomes Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Sleep, mood, cognition, and quality-of-life PRO instruments — PSQI, PHQ-9, GAD-7, QOLIE-31, MoCA, WPAI
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
          <Card title="Total Patients"><KPI value={ov.total_patients} label="Patients assessed" color="#3b82f6" /></Card>
          <Card title="Total Assessments"><KPI value={ov.total_assessments} label="PRO assessments" color="#8b5cf6" /></Card>
          <Card title="Avg PSQI"><KPI value={ov.avg_psqi != null ? ov.avg_psqi.toFixed(1) : '--'} label="Sleep quality" color="#06b6d4" /></Card>
          <Card title="Avg PHQ-9"><KPI value={ov.avg_phq9 != null ? ov.avg_phq9.toFixed(1) : '--'} label="Depression severity" color="#ef4444" /></Card>

          {/* KPI Row 2 */}
          <Card title="Avg GAD-7"><KPI value={ov.avg_gad7 != null ? ov.avg_gad7.toFixed(1) : '--'} label="Anxiety severity" color="#f59e0b" /></Card>
          <Card title="Avg QOLIE-31"><KPI value={ov.avg_qolie31 != null ? ov.avg_qolie31.toFixed(1) : '--'} label="Quality of life" color="#10b981" /></Card>
          <Card title="Avg MoCA"><KPI value={ov.avg_moca != null ? ov.avg_moca.toFixed(1) : '--'} label="Cognitive score" color="#ec4899" /></Card>
          <Card title="Avg WPAI"><KPI value={ov.avg_wpai != null ? ov.avg_wpai.toFixed(1) : '--'} label="Work productivity" color="#f97316" /></Card>

          {/* Depression Severity Bar Chart */}
          <Card title="Depression Severity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.depression_severity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(ov.depression_severity || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Anxiety Severity Bar Chart */}
          <Card title="Anxiety Severity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.anxiety_severity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(ov.anxiety_severity || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Sleep Quality Distribution Pie Chart */}
          <Card title="Sleep Quality Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.sleep_quality_distribution || []} dataKey="count" nameKey="quality" cx="50%" cy="50%" outerRadius={80} label={e => `${e.quality}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.sleep_quality_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* QoL Trend Line Chart */}
          <Card title="Quality of Life Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={ov.qol_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" fontSize={10} angle={-30} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_score" name="Avg QOLIE-31" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Domain Averages Bar Chart */}
          <Card title="PRO Domain Averages" span={4}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.domain_averages || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="average" name="Average Score" radius={[4, 4, 0, 0]}>
                  {(ov.domain_averages || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* SLEEP & FATIGUE TAB */}
      {tab === 'sleep_fatigue' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Sleep KPIs */}
          <Card title="Avg PSQI Score"><KPI value={ov.avg_psqi != null ? ov.avg_psqi.toFixed(1) : '--'} label="Pittsburgh Sleep Quality Index" color="#06b6d4" /></Card>
          <Card title="Avg ESS Score"><KPI value={ov.avg_ess != null ? ov.avg_ess.toFixed(1) : '--'} label="Epworth Sleepiness Scale" color="#8b5cf6" /></Card>

          {/* Sleep Quality Pie Chart */}
          <Card title="Sleep Quality Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={ov.sleep_quality_distribution || []} dataKey="count" nameKey="quality" cx="50%" cy="50%" outerRadius={90} label={e => `${e.quality}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.sleep_quality_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Sleep Data Table */}
          <Card title="Sleep Assessment Details" span={2}>
            {(bd.sleep_data || []).length === 0 ? (
              <p style={{ color: '#94a3b8', fontSize: 12 }}>No sleep assessment data available.</p>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      {['Patient ID', 'PSQI', 'ESS', 'Sleep Quality', 'Sleep Duration', 'Assessment Date'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.sleep_data || []).map((s, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(s.patient_id)}</td>
                        <td style={{ padding: '6px' }}>{fmt(s.psqi)}</td>
                        <td style={{ padding: '6px' }}>{fmt(s.ess)}</td>
                        <td style={{ padding: '6px' }}>{fmt(s.sleep_quality)}</td>
                        <td style={{ padding: '6px' }}>{fmt(s.sleep_duration)}</td>
                        <td style={{ padding: '6px' }}>{fmt(s.assessment_date)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* MOOD & ANXIETY TAB */}
      {tab === 'mood_anxiety' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* Mood KPIs */}
          <Card title="Avg PHQ-9"><KPI value={ov.avg_phq9 != null ? ov.avg_phq9.toFixed(1) : '--'} label="Depression severity" color="#ef4444" /></Card>
          <Card title="Avg GAD-7"><KPI value={ov.avg_gad7 != null ? ov.avg_gad7.toFixed(1) : '--'} label="Anxiety severity" color="#f59e0b" /></Card>
          <Card title="Avg NDDI-E"><KPI value={ov.avg_nddi_e != null ? ov.avg_nddi_e.toFixed(1) : '--'} label="Epilepsy depression" color="#ec4899" /></Card>

          {/* Depression Severity Bar Chart */}
          <Card title="Depression Severity (PHQ-9)" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov.depression_severity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(ov.depression_severity || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Mood vs Seizure Worry */}
          <Card title="Mood vs Seizure Worry">
            {bd.mood_vs_seizure_worry ? (
              <div style={{ padding: 12 }}>
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Correlation Coefficient</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: bd.mood_vs_seizure_worry.correlation > 0 ? '#e74c3c' : '#2ecc71' }}>
                    {bd.mood_vs_seizure_worry.correlation != null ? bd.mood_vs_seizure_worry.correlation.toFixed(3) : '--'}
                  </div>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Interpretation</div>
                  <div style={{ fontSize: 13, color: '#1e293b' }}>{bd.mood_vs_seizure_worry.interpretation || '--'}</div>
                </div>
                {bd.mood_vs_seizure_worry.p_value != null && (
                  <div>
                    <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>P-Value</div>
                    <div style={{ fontSize: 13, color: '#1e293b' }}>{bd.mood_vs_seizure_worry.p_value.toFixed(4)}</div>
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: '#94a3b8', fontSize: 12 }}>No correlation data available.</p>
            )}
          </Card>

          {/* Anxiety Severity Bar Chart */}
          <Card title="Anxiety Severity (GAD-7)" span={3}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov.anxiety_severity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(ov.anxiety_severity || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patient_detail' && (
        <Card title="Patient PRO Summary">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient ID', 'Total Assessments', 'Latest PHQ-9', 'Latest QOLIE-31', 'Latest MoCA', 'Trend'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(p.patient_id)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.total_assessments)}</td>
                    <td style={{ padding: '6px', fontWeight: 600, color: '#ef4444' }}>{fmt(p.latest_phq9)}</td>
                    <td style={{ padding: '6px', fontWeight: 600, color: '#10b981' }}>{fmt(p.latest_qolie31)}</td>
                    <td style={{ padding: '6px', fontWeight: 600, color: '#ec4899' }}>{fmt(p.latest_moca)}</td>
                    <td style={{ padding: '6px', fontSize: 18 }}>
                      {p.trend === 'improving' && <Badge text="Improving" color="#2ecc71" />}
                      {p.trend === 'stable' && <Badge text="Stable" color="#3b82f6" />}
                      {p.trend === 'declining' && <Badge text="Declining" color="#e74c3c" />}
                      {!p.trend && '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="PRO Outcome Concepts">
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
