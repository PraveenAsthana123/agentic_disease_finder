import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

const TRAJ_COLORS = {
  improving: '#10b981',
  stable: '#3b82f6',
  declining: '#ef4444',
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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'timelines', label: 'Patient Timelines' },
  { id: 'domains', label: 'Domain Scores' },
  { id: 'comorbidities', label: 'Comorbidity Flags' },
  { id: 'definitions', label: 'Definitions' },
]

export default function FunctionalRecoveryDashboard() {
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
          axios.get(`${API_URL}/api/functional-recovery/overview`),
          axios.get(`${API_URL}/api/functional-recovery/breakdown`),
          axios.get(`${API_URL}/api/functional-recovery/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load functional recovery data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      Loading functional recovery data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )

  const k = overview?.kpis || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="KPI Summary" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 12 }}>
          <KPI label="Total Patients" value={k.total_patients} />
          <KPI label="Total Assessments" value={k.total_assessments} />
          <KPI label="Avg Daily Function" value={fmt(k.avg_daily_function)} sub="/ 10" />
          <KPI label="Avg Social Function" value={fmt(k.avg_social_function)} sub="/ 10" />
          <KPI label="Avg QOLIE-31" value={fmt(k.avg_qolie31)} sub="/ 100" />
          <KPI label="Avg WPAI" value={fmt(k.avg_wpai)} sub="% impaired" color={k.avg_wpai > 40 ? '#f59e0b' : '#10b981'} />
          <KPI label="Improving" value={k.patients_improving} color="#10b981" />
          <KPI label="Declining" value={k.patients_declining} color="#ef4444" />
        </div>
      </Card>

      <Card title="Function Trend Over Time" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={overview?.function_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 10]} />
            <Tooltip />
            <Line type="monotone" dataKey="avg_daily" name="Daily Function" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="avg_social" name="Social Function" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Quality of Life Distribution" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview?.qolie_distribution || []} dataKey="count" nameKey="tier"
              cx="50%" cy="50%" outerRadius={80} label={({ tier, count }) => `${tier}: ${count}`}>
              {(overview?.qolie_distribution || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Fatigue Distribution" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview?.fatigue_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="level" tick={{ fontSize: 12 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {(overview?.fatigue_distribution || []).map((entry, i) => (
                <Cell key={i} fill={entry.level === 'High' ? '#ef4444' : entry.level === 'Moderate' ? '#f59e0b' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Recovery Trajectories" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>First Daily</th>
                <th style={{ padding: '8px 12px' }}>Last Daily</th>
                <th style={{ padding: '8px 12px' }}>Change</th>
                <th style={{ padding: '8px 12px' }}>Trajectory</th>
              </tr>
            </thead>
            <tbody>
              {(overview?.recovery_summary || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(r.first_daily)}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(r.last_daily)}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 600,
                    color: r.change > 0 ? '#10b981' : r.change < 0 ? '#ef4444' : '#64748b'
                  }}>{r.change > 0 ? '+' : ''}{fmt(r.change)}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <Badge text={r.trajectory} color={TRAJ_COLORS[r.trajectory] || '#64748b'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  const renderTimelines = () => {
    const pts = breakdown?.patient_timelines || []
    return (
      <div style={{ display: 'grid', gap: 16 }}>
        {pts.map((p, i) => (
          <Card key={i} title={
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{p.patient_id} — {p.assessment_count} assessments</span>
              <Badge text={p.trajectory} color={TRAJ_COLORS[p.trajectory] || '#64748b'} />
            </div>
          }>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 12 }}>
              <KPI label="Latest Daily" value={fmt(p.latest_daily)} sub="/ 10" />
              <KPI label="Latest Social" value={fmt(p.latest_social)} sub="/ 10" />
              <KPI label="Assessments" value={p.assessment_count} />
              <KPI label="Trajectory" value={p.trajectory}
                color={TRAJ_COLORS[p.trajectory] || '#64748b'} />
            </div>
            {p.assessments && p.assessments.length > 0 && (
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={p.assessments}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 9 }} />
                  <YAxis domain={[0, 10]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="daily_function" name="Daily" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="social_function" name="Social" stroke="#10b981" strokeWidth={2} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            )}
            {p.assessments && p.assessments.length > 0 && (
              <div style={{ overflowX: 'auto', marginTop: 10 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '6px 8px' }}>Date</th>
                      <th style={{ padding: '6px 8px' }}>Daily</th>
                      <th style={{ padding: '6px 8px' }}>Social</th>
                      <th style={{ padding: '6px 8px' }}>MoCA</th>
                      <th style={{ padding: '6px 8px' }}>QOLIE</th>
                      <th style={{ padding: '6px 8px' }}>WPAI%</th>
                      <th style={{ padding: '6px 8px' }}>Fatigue</th>
                      <th style={{ padding: '6px 8px' }}>Mood</th>
                      <th style={{ padding: '6px 8px' }}>Sleep</th>
                      <th style={{ padding: '6px 8px' }}>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.assessments.map((a, j) => (
                      <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{a.date}</td>
                        <td style={{ padding: '6px 8px' }}>{fmt(a.daily_function)}</td>
                        <td style={{ padding: '6px 8px' }}>{fmt(a.social_function)}</td>
                        <td style={{ padding: '6px 8px', color: a.moca < 22 ? '#ef4444' : undefined }}>{fmt(a.moca)}</td>
                        <td style={{ padding: '6px 8px' }}>{fmt(a.qolie31)}</td>
                        <td style={{ padding: '6px 8px', color: a.wpai > 50 ? '#ef4444' : undefined }}>{fmt(a.wpai)}</td>
                        <td style={{ padding: '6px 8px' }}>{fmt(a.fatigue)}</td>
                        <td style={{ padding: '6px 8px' }}>{fmt(a.mood)}</td>
                        <td style={{ padding: '6px 8px' }}>{fmt(a.sleep_hours)}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, maxWidth: 200, color: '#64748b' }}>{(a.notes || '').slice(0, 80)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        ))}
      </div>
    )
  }

  const renderDomains = () => {
    const ds = breakdown?.domain_scores || []
    const mv = breakdown?.monthly_volume || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Functional Domain Scores" span={1}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Domain</th>
                  <th style={{ padding: '8px 12px' }}>Avg Score</th>
                  <th style={{ padding: '8px 12px' }}>Below Threshold</th>
                  <th style={{ padding: '8px 12px' }}>Threshold</th>
                </tr>
              </thead>
              <tbody>
                {ds.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{d.domain}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(d.avg_score)}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge text={`${d.patients_below_threshold} patients`}
                        color={d.patients_below_threshold > 0 ? '#ef4444' : '#10b981'} />
                    </td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{d.threshold}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Domain Averages" span={1}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={ds} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="domain" type="category" width={130} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="avg_score" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Monthly Assessment Volume" span={2}>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={mv}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    )
  }

  const renderComorbidities = () => {
    const flags = breakdown?.comorbidity_flags || []
    return (
      <Card title={`Comorbidity Flags (${flags.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Memory Complaints</th>
                <th style={{ padding: '8px 12px' }}>Concentration Difficulty</th>
                <th style={{ padding: '8px 12px' }}>Latest PHQ-9</th>
                <th style={{ padding: '8px 12px' }}>Latest GAD-7</th>
                <th style={{ padding: '8px 12px' }}>Flag Count</th>
              </tr>
            </thead>
            <tbody>
              {flags.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9',
                  background: f.flag_count >= 3 ? '#fef2f2' : undefined }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{f.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>
                    {f.has_memory_complaints
                      ? <Badge text="Yes" color="#ef4444" />
                      : <span style={{ color: '#94a3b8' }}>No</span>}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    {f.has_concentration_difficulty
                      ? <Badge text="Yes" color="#f59e0b" />
                      : <span style={{ color: '#94a3b8' }}>No</span>}
                  </td>
                  <td style={{ padding: '8px 12px', color: f.latest_phq9 >= 10 ? '#ef4444' : undefined }}>
                    {fmt(f.latest_phq9)}
                  </td>
                  <td style={{ padding: '8px 12px', color: f.latest_gad7 >= 10 ? '#ef4444' : undefined }}>
                    {fmt(f.latest_gad7)}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <Badge text={f.flag_count}
                      color={f.flag_count >= 3 ? '#ef4444' : f.flag_count >= 2 ? '#f59e0b' : '#10b981'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )
  }

  const renderDefinitions = () => {
    if (!defs) return null
    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title="Functional Recovery Concepts">
          <div style={{ display: 'grid', gap: 10 }}>
            {(defs.concepts || []).map((c, i) => (
              <div key={i} style={{ padding: '10px 14px', background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.term}</div>
                <div style={{ fontSize: 13, color: '#475569' }}>{c.definition}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Domain Thresholds">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Domain</th>
                  <th style={{ padding: '8px 12px' }}>Threshold</th>
                  <th style={{ padding: '8px 12px' }}>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {(defs.thresholds || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{t.domain}</td>
                    <td style={{ padding: '8px 12px' }}><Badge text={t.threshold} color="#3b82f6" /></td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>{t.interpretation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Quality Metrics">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Metric</th>
                  <th style={{ padding: '8px 12px' }}>Target</th>
                  <th style={{ padding: '8px 12px' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.quality_metrics || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{m.metric}</td>
                    <td style={{ padding: '8px 12px' }}><Badge text={m.target} color="#3b82f6" /></td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>{m.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Compliance References">
          <div style={{ display: 'grid', gap: 8 }}>
            {(defs.compliance_references || []).map((r, i) => (
              <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 13 }}>
                <span style={{ fontWeight: 600, color: '#1e293b' }}>{r.ref}</span>
                <span style={{ color: '#64748b', marginLeft: 8 }}>— {r.note}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Functional Recovery Tracker</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Longitudinal functional recovery monitoring — daily/social function, QoL, cognition, work productivity
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'timelines' && renderTimelines()}
      {tab === 'domains' && renderDomains()}
      {tab === 'comorbidities' && renderComorbidities()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
