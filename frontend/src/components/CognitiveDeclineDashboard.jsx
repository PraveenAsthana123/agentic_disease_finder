import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'

const CLASS_COLORS = {
  stable: '#22c55e', improving: '#3b82f6',
  mild_decline: '#f59e0b', moderate_decline: '#f97316', significant_decline: '#ef4444'
}
const RISK_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#22c55e' }
const DOMAIN_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']

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

function Badge({ label, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function CognitiveDeclineDashboard() {
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
          axios.get(`${API_URL}/cognitive-decline/overview`),
          axios.get(`${API_URL}/cognitive-decline/breakdown`),
          axios.get(`${API_URL}/cognitive-decline/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading cognitive decline data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'trends', label: 'Patient Trends' },
    { id: 'risk', label: 'Risk Stratification' },
    { id: 'alerts', label: 'Alerts' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Cognitive Decline Tracker</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Longitudinal cognitive assessment tracking — early detection of cognitive decline in epilepsy patients
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'trends' && <TrendsTab data={breakdown} />}
      {tab === 'risk' && <RiskTab data={breakdown} />}
      {tab === 'alerts' && <AlertsTab data={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}


function OverviewTab({ data }) {
  if (!data) return null
  const k = data.kpis

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      <Card span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          <KPI label="Patients Tracked" value={fmt(k.total_patients)} />
          <KPI label="With Decline" value={fmt(k.patients_with_decline)} color="#ef4444" />
          <KPI label="Decline Rate" value={`${fmt(k.decline_rate_pct)}%`} color={k.decline_rate_pct > 30 ? '#ef4444' : '#f59e0b'} />
          <KPI label="Avg Follow-up" value={`${fmt(k.avg_follow_up_months)} mo`} />
          <KPI label="Total Assessments" value={fmt(k.total_assessments)} />
        </div>
      </Card>

      <Card title="Decline Classification">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={data.classification_distribution} dataKey="count" nameKey="label"
                 cx="50%" cy="50%" outerRadius={80}
                 label={({ label, count }) => `${label}: ${count}`}>
              {data.classification_distribution.map((d, i) => (
                <Cell key={i} fill={d.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Domain Decline Rates">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.domain_decline_rates}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="domain" tick={{ fontSize: 11 }} />
            <YAxis label={{ value: '%', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Bar dataKey="rate_pct" radius={[6, 6, 0, 0]}>
              {data.domain_decline_rates.map((_, i) => (
                <Cell key={i} fill={DOMAIN_COLORS[i % DOMAIN_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="MoCA Trend (Avg by Timepoint)">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data.moca_trend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timepoint" label={{ value: 'Timepoint', position: 'insideBottom', offset: -5 }} />
            <YAxis domain={['auto', 'auto']} label={{ value: 'MoCA', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(v) => v.toFixed(1)} />
            <Line type="monotone" dataKey="avg_moca" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Risk Stratification" span={2}>
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', justifyContent: 'center', padding: '20px 0' }}>
          {data.risk_stratification.map(r => (
            <div key={r.level} style={{ textAlign: 'center' }}>
              <div style={{
                width: 80, height: 80, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: r.color + '18', border: `3px solid ${r.color}`, margin: '0 auto 8px'
              }}>
                <span style={{ fontSize: 28, fontWeight: 700, color: r.color }}>{r.count}</span>
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: r.color, textTransform: 'capitalize' }}>{r.level} Risk</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Classification Summary">
        {data.classification_distribution.map(c => (
          <div key={c.classification} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '6px 0', borderBottom: '1px solid #f1f5f9'
          }}>
            <Badge label={c.label} color={c.color} />
            <span style={{ fontWeight: 600, fontSize: 14 }}>{c.count}</span>
          </div>
        ))}
      </Card>
    </div>
  )
}


function TrendsTab({ data }) {
  if (!data) return null
  const [selected, setSelected] = useState(null)
  const patients = data.patients

  if (selected) {
    const p = patients.find(x => x.patient_id === selected)
    if (!p) return <div>Patient not found</div>
    return (
      <div>
        <button onClick={() => setSelected(null)} style={{
          padding: '6px 14px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff',
          cursor: 'pointer', fontSize: 13, marginBottom: 16
        }}>Back to list</button>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title={`${p.name} (${p.patient_id})`} span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12, marginBottom: 16 }}>
              <KPI label="Classification" value={p.classification_label} color={p.classification_color} />
              <KPI label="Risk Level" value={p.risk_level.charAt(0).toUpperCase() + p.risk_level.slice(1)} color={p.risk_color} />
              <KPI label="MoCA Slope" value={`${p.moca_slope > 0 ? '+' : ''}${p.moca_slope}`} sub="pts/year" color={p.moca_slope < -1 ? '#ef4444' : p.moca_slope > 1 ? '#3b82f6' : '#22c55e'} />
              <KPI label="MoCA Change" value={`${p.moca_change > 0 ? '+' : ''}${p.moca_change}`} sub={`${p.baseline_moca} → ${p.latest_moca}`} />
              <KPI label="Follow-up" value={`${p.follow_up_months} mo`} sub={`${p.n_timepoints} timepoints`} />
            </div>
          </Card>

          <Card title="MoCA & MMSE Trajectory">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={p.timepoints}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="months" label={{ value: 'Months', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={[0, 30]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="moca" stroke="#3b82f6" strokeWidth={2} name="MoCA" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="mmse" stroke="#8b5cf6" strokeWidth={2} name="MMSE" dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Domain Indices Trajectory">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={p.timepoints}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="months" label={{ value: 'Months', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="memory" stroke="#3b82f6" strokeWidth={2} name="Memory" dot={{ r: 3 }} />
                <Line type="monotone" dataKey="attention" stroke="#8b5cf6" strokeWidth={2} name="Attention" dot={{ r: 3 }} />
                <Line type="monotone" dataKey="executive" stroke="#f59e0b" strokeWidth={2} name="Executive" dot={{ r: 3 }} />
                <Line type="monotone" dataKey="processing_speed" stroke="#ef4444" strokeWidth={2} name="Processing Speed" dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Assessment Timeline" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: 6 }}>TP</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>Date</th>
                    <th style={{ padding: 6 }}>Months</th>
                    <th style={{ padding: 6 }}>MoCA</th>
                    <th style={{ padding: 6 }}>MMSE</th>
                    <th style={{ padding: 6 }}>Memory</th>
                    <th style={{ padding: 6 }}>Attention</th>
                    <th style={{ padding: 6 }}>Executive</th>
                    <th style={{ padding: 6 }}>Speed</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {p.timepoints.map((tp, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 6 }}>{tp.timepoint}</td>
                      <td style={{ padding: 6 }}>{tp.date}</td>
                      <td style={{ padding: 6, textAlign: 'center' }}>{fmt(tp.months)}</td>
                      <td style={{ padding: 6, textAlign: 'center', fontWeight: 600 }}>{fmt(tp.moca)}</td>
                      <td style={{ padding: 6, textAlign: 'center' }}>{fmt(tp.mmse)}</td>
                      <td style={{ padding: 6, textAlign: 'center' }}>{fmt(tp.memory)}</td>
                      <td style={{ padding: 6, textAlign: 'center' }}>{fmt(tp.attention)}</td>
                      <td style={{ padding: 6, textAlign: 'center' }}>{fmt(tp.executive)}</td>
                      <td style={{ padding: 6, textAlign: 'center' }}>{fmt(tp.processing_speed)}</td>
                      <td style={{ padding: 6, fontSize: 11, color: '#64748b' }}>{tp.notes || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <Card title="Patient Cognitive Trajectories">
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
            <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
            <th style={{ padding: 8 }}>Timepoints</th>
            <th style={{ padding: 8 }}>Baseline MoCA</th>
            <th style={{ padding: 8 }}>Latest MoCA</th>
            <th style={{ padding: 8 }}>Slope</th>
            <th style={{ padding: 8 }}>Classification</th>
            <th style={{ padding: 8 }}>Risk</th>
            <th style={{ padding: 8 }}></th>
          </tr>
        </thead>
        <tbody>
          {patients.map(p => (
            <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: 8 }}>{p.name}<br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{p.patient_id}</span></td>
              <td style={{ padding: 8, textAlign: 'center' }}>{p.n_timepoints}</td>
              <td style={{ padding: 8, textAlign: 'center' }}>{fmt(p.baseline_moca)}</td>
              <td style={{ padding: 8, textAlign: 'center' }}>{fmt(p.latest_moca)}</td>
              <td style={{ padding: 8, textAlign: 'center', fontWeight: 600, color: p.moca_slope < -1 ? '#ef4444' : p.moca_slope > 1 ? '#3b82f6' : '#22c55e' }}>
                {p.moca_slope > 0 ? '+' : ''}{p.moca_slope}
              </td>
              <td style={{ padding: 8, textAlign: 'center' }}>
                <Badge label={p.classification_label} color={p.classification_color} />
              </td>
              <td style={{ padding: 8, textAlign: 'center' }}>
                <Badge label={p.risk_level.charAt(0).toUpperCase() + p.risk_level.slice(1)} color={p.risk_color} />
              </td>
              <td style={{ padding: 8 }}>
                <button onClick={() => setSelected(p.patient_id)} style={{
                  padding: '4px 10px', borderRadius: 6, border: '1px solid #e2e8f0',
                  background: '#f8fafc', cursor: 'pointer', fontSize: 12
                }}>View</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}


function RiskTab({ data }) {
  if (!data) return null
  const patients = data.patients

  const groups = { high: [], medium: [], low: [] }
  patients.forEach(p => {
    groups[p.risk_level].push(p)
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {['high', 'medium', 'low'].map(level => (
        <Card key={level} title={`${level.charAt(0).toUpperCase() + level.slice(1)} Risk (${groups[level].length})`}>
          {groups[level].length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 13 }}>No patients in this category</p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                  <th style={{ padding: 8 }}>Age</th>
                  <th style={{ padding: 8 }}>MoCA (Baseline/Latest)</th>
                  <th style={{ padding: 8 }}>MoCA Slope</th>
                  <th style={{ padding: 8 }}>Classification</th>
                  <th style={{ padding: 8 }}>Declining Domains</th>
                </tr>
              </thead>
              <tbody>
                {groups[level].map(p => {
                  const decliningDomains = Object.entries(p.domain_slopes)
                    .filter(([, v]) => v < -3.0)
                    .map(([d]) => d)
                  return (
                    <tr key={p.patient_id} style={{
                      borderBottom: '1px solid #f1f5f9',
                      background: level === 'high' ? '#fef2f2' : level === 'medium' ? '#fffbeb' : undefined
                    }}>
                      <td style={{ padding: 8 }}>{p.name}<br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{p.patient_id}</span></td>
                      <td style={{ padding: 8, textAlign: 'center' }}>{p.age || '--'}</td>
                      <td style={{ padding: 8, textAlign: 'center' }}>{fmt(p.baseline_moca)} / {fmt(p.latest_moca)}</td>
                      <td style={{ padding: 8, textAlign: 'center', fontWeight: 600, color: p.moca_slope < -1 ? '#ef4444' : '#22c55e' }}>
                        {p.moca_slope > 0 ? '+' : ''}{p.moca_slope}
                      </td>
                      <td style={{ padding: 8, textAlign: 'center' }}>
                        <Badge label={p.classification_label} color={p.classification_color} />
                      </td>
                      <td style={{ padding: 8, textAlign: 'center', fontSize: 12 }}>
                        {decliningDomains.length > 0 ? decliningDomains.join(', ') : <span style={{ color: '#94a3b8' }}>None</span>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </Card>
      ))}
    </div>
  )
}


function AlertsTab({ data }) {
  if (!data) return null
  const flagged = data.flagged_patients

  return (
    <Card title={`Clinical Alerts (${flagged.length} patients)`}>
      <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 12px' }}>
        Patients flagged for clinical attention due to significant or moderate cognitive decline, or high risk classification
      </p>
      {flagged.length === 0 ? (
        <p style={{ color: '#22c55e', fontWeight: 500 }}>No patients currently flagged</p>
      ) : (
        <div style={{ display: 'grid', gap: 12 }}>
          {flagged.map(f => (
            <div key={f.patient_id} style={{
              padding: 16, borderRadius: 8, border: `1px solid ${f.classification_color}33`,
              background: f.classification_color + '08'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div>
                  <span style={{ fontWeight: 600, fontSize: 14 }}>{f.name}</span>
                  <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 8 }}>{f.patient_id}</span>
                  {f.age && <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 8 }}>Age: {f.age}</span>}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <Badge label={f.classification} color={f.classification_color} />
                  <Badge label={`${f.risk_level.charAt(0).toUpperCase() + f.risk_level.slice(1)} Risk`} color={RISK_COLORS[f.risk_level]} />
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 8, fontSize: 12 }}>
                <div><span style={{ color: '#64748b' }}>MoCA Slope:</span> <b style={{ color: '#ef4444' }}>{f.moca_slope} pts/yr</b></div>
                <div><span style={{ color: '#64748b' }}>MoCA Change:</span> <b>{f.moca_change > 0 ? '+' : ''}{f.moca_change}</b></div>
                <div><span style={{ color: '#64748b' }}>Follow-up:</span> <b>{f.follow_up_months} months</b></div>
                <div><span style={{ color: '#64748b' }}>Declining Domains:</span> <b>{f.declining_domains.length > 0 ? f.declining_domains.join(', ') : 'None'}</b></div>
              </div>
              <div style={{ fontSize: 12, padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
                <span style={{ fontWeight: 600, color: '#334155' }}>Recommendation:</span>{' '}
                <span style={{ color: '#475569' }}>{f.recommendation}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}


function DefinitionsTab({ data }) {
  if (!data) return null

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Decline Classification Thresholds" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {data.decline_thresholds.map(t => (
            <div key={t.classification} style={{ padding: 12, background: t.color + '08', borderRadius: 8, borderLeft: `4px solid ${t.color}` }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: t.color, marginBottom: 4 }}>{t.label}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>MoCA Slope: {t.moca_slope}</div>
              <div style={{ fontSize: 12, color: '#475569' }}>{t.description}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Assessment Instruments" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {data.assessment_instruments.map(inst => (
            <div key={inst.name} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{inst.name}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>{inst.description}</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                Range: {inst.range} | Cutoff: {inst.cutoff}
                {inst.time && ` | Time: ${inst.time}`}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Risk Factors for Cognitive Decline in Epilepsy">
        {data.risk_factors.map(rf => (
          <div key={rf.factor} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
            <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{rf.factor}:</span>{' '}
            <span style={{ fontSize: 12, color: '#475569' }}>{rf.description}</span>
          </div>
        ))}
      </Card>

      <Card title="Clinical Action Recommendations">
        {data.clinical_actions.map(ca => (
          <div key={ca.category} style={{ padding: 12, marginBottom: 8, background: ca.color + '08', borderRadius: 8, borderLeft: `4px solid ${ca.color}` }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: ca.color, marginBottom: 6 }}>{ca.category}</div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569' }}>
              {ca.actions.map((a, i) => <li key={i} style={{ marginBottom: 2 }}>{a}</li>)}
            </ul>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>Follow-up: {ca.follow_up}</div>
          </div>
        ))}
      </Card>

      <Card title="Glossary" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {data.glossary.map(g => (
            <div key={g.term} style={{ padding: 8, background: '#f8fafc', borderRadius: 6 }}>
              <span style={{ fontWeight: 600, fontSize: 13 }}>{g.term}:</span>{' '}
              <span style={{ fontSize: 12, color: '#475569' }}>{g.definition}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
