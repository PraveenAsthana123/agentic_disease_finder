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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const SEVERITY_COLORS = {
  normal: '#10b981',
  mild: '#f59e0b',
  moderate: '#f97316',
  severe: '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'assessments', label: 'Assessment Inventory' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'alerts', label: 'Clinical Alerts' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function VoiceAIDashboard() {
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
      axios.get(`${API_URL}/api/voice-ai/overview`),
      axios.get(`${API_URL}/api/voice-ai/breakdown`),
      axios.get(`${API_URL}/api/voice-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Voice AI data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No voice AI data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Voice AI Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Vocal biomarker analysis, articulation/fluency/swallowing assessments (WAB, Verbal Fluency, MASA, BNT, Digit Span)
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'assessments' && <AssessmentsTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'alerts' && <AlertsTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = overview.kpis || []
  const instDist = overview.instrument_distribution || []
  const sevDist = overview.severity_distribution || []
  const instScores = overview.instrument_scores || []
  const dailyActivity = overview.daily_activity || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color || COLORS[i % COLORS.length]} /></Card>
      ))}

      {/* Instrument Distribution Pie */}
      <Card title="Assessment Instrument Distribution" span={2}>
        {instDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={instDist} dataKey="count" nameKey="label" cx="50%" cy="50%" outerRadius={100}
                label={({ label, count }) => `${label} (${count})`}>
                {instDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No instrument data</div>}
      </Card>

      {/* Severity Distribution Bar */}
      <Card title="Severity Distribution" span={2}>
        {sevDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={sevDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {sevDist.map((entry, i) => (
                  <Cell key={i} fill={entry.color || COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No severity data</div>}
      </Card>

      {/* Per-Instrument Mean Score Bar */}
      <Card title="Mean Score by Instrument" span={2}>
        {instScores.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={instScores} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <YAxis dataKey="label" type="category" tick={{ fontSize: 10 }} width={160} />
              <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
              <Bar dataKey="mean_score" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No instrument score data</div>}
      </Card>

      {/* Daily Activity Line Chart */}
      <Card title="Daily Assessment Activity" span={2}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily activity data</div>}
      </Card>
    </div>
  )
}

function AssessmentsTab({ breakdown }) {
  const inventory = breakdown?.assessment_inventory || []
  const instStats = breakdown?.instrument_stats || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Instrument Statistics */}
      <Card title={`Instrument Statistics (${instStats.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Instrument', 'Domain', 'Count', 'Mean Score', 'Normal', 'Mild', 'Moderate', 'Severe', 'Alerts'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Instrument' || h === 'Domain' ? 'left' : 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {instStats.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.label}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{s.domain}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{s.count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    <Badge text={s.mean_score != null ? `${(s.mean_score * 100).toFixed(1)}%` : 'N/A'}
                      color={s.mean_score >= 0.8 ? '#10b981' : s.mean_score >= 0.6 ? '#f59e0b' : '#ef4444'} />
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#10b981' }}>{s.normal || 0}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#f59e0b' }}>{s.mild || 0}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#f97316' }}>{s.moderate || 0}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#ef4444' }}>{s.severe || 0}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    {s.alerts > 0 ? <Badge text={s.alerts} color="#ef4444" /> : <span style={{ color: '#94a3b8' }}>0</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Full Assessment Inventory */}
      <Card title={`Assessment Inventory (${inventory.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['ID', 'Patient', 'Instrument', 'Domain', 'Score', 'Level', 'Interpretation', 'Date'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Score' ? 'right' : 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {inventory.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={a.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.instrument_label}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{a.domain}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    {a.score != null ? `${a.score}/${a.max_score}` : 'N/A'}
                    {a.pct != null && <span style={{ color: '#94a3b8', marginLeft: 4 }}>({a.pct}%)</span>}
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    {a.level && <Badge text={a.level} color={SEVERITY_COLORS[a.level] || '#6b7280'} />}
                  </td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {a.interpretation || ''}
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{a.created_at ? a.created_at.slice(0, 10) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const patients = breakdown?.patient_profiles || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {patients.map((p, i) => (
        <Card key={i} title={`Patient ${p.patient_id}${p.name ? ` — ${p.name}` : ''}`}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
            {p.age && <div><span style={{ fontSize: 12, color: '#64748b' }}>Age: </span><strong>{p.age}</strong></div>}
            {p.gender && <div><span style={{ fontSize: 12, color: '#64748b' }}>Gender: </span><strong>{p.gender}</strong></div>}
            {p.disease && <div><span style={{ fontSize: 12, color: '#64748b' }}>Disease: </span><Badge text={p.disease} color="#8b5cf6" /></div>}
          </div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Assessments: </span><strong>{p.n_assessments}</strong></div>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Mean Score: </span>
              <Badge text={p.mean_score != null ? `${(p.mean_score * 100).toFixed(1)}%` : 'N/A'}
                color={p.mean_score >= 0.8 ? '#10b981' : p.mean_score >= 0.6 ? '#f59e0b' : '#ef4444'} />
            </div>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Worst Level: </span>
              <Badge text={p.worst_level || 'N/A'} color={SEVERITY_COLORS[p.worst_level] || '#6b7280'} />
            </div>
            {p.alert_count > 0 && (
              <div><span style={{ fontSize: 12, color: '#64748b' }}>Alerts: </span>
                <Badge text={p.alert_count} color="#ef4444" />
              </div>
            )}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {(p.instrument_labels || []).map((inst, j) => (
              <Badge key={j} text={inst} color={COLORS[j % COLORS.length]} />
            ))}
          </div>
        </Card>
      ))}
      {patients.length === 0 && <Card span={2}><div style={{ color: '#94a3b8', fontSize: 13 }}>No patient data</div></Card>}
    </div>
  )
}

function AlertsTab({ breakdown }) {
  const alerts = breakdown?.clinical_alerts || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Clinical Alerts (${alerts.length})`}>
        {alerts.length > 0 ? (
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#fef2f2' }}>
                  {['ID', 'Patient', 'Instrument', 'Level', 'Score', 'Alert', 'Date'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Score' ? 'right' : 'left', borderBottom: '1px solid #fecaca', color: '#991b1b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {alerts.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #fef2f2' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.id}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={a.patient_id} color="#3b82f6" /></td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.instrument}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <Badge text={a.level || 'N/A'} color={SEVERITY_COLORS[a.level] || '#6b7280'} />
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                      {a.score != null ? `${a.score}/${a.max_score}` : 'N/A'}
                    </td>
                    <td style={{ padding: '6px 10px', color: '#991b1b', fontWeight: 500 }}>{a.alert}</td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{a.created_at ? a.created_at.slice(0, 10) : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#10b981', fontSize: 13, padding: 20, textAlign: 'center' }}>No clinical alerts — all assessments within normal range</div>}
      </Card>
    </div>
  )
}

function PipelineTab({ breakdown }) {
  const events = breakdown?.pipeline_events || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Voice Pipeline Events (${events.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['ID', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{e.id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={e.component} color="#8b5cf6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{e.action}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{e.actor || 'system'}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 350, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || ''}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{e.ts_local || e.ts_utc || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div></Card>

  const concepts = definitions.concepts || []
  const qualityMetrics = definitions.quality_metrics || []
  const instruments = definitions.assessment_instruments || []
  const compliance = definitions.compliance || []
  const remediation = definitions.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Voice AI Concepts */}
      <Card title={`Voice AI Concepts (${concepts.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Term</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Assessment Instruments */}
      <Card title={`Assessment Instruments (${instruments.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Instrument</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Full Name</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Domain</th>
            </tr>
          </thead>
          <tbody>
            {instruments.map((inst, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600 }}><Badge text={inst.instrument} color={COLORS[i % COLORS.length]} /></td>
                <td style={{ padding: '8px 12px' }}>{inst.label}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{inst.domain}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Quality Metrics */}
      <Card title={`Quality Metrics (${qualityMetrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {qualityMetrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Compliance */}
      <Card title={`Compliance References (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Standard</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Notes</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.ref}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Remediation */}
      <Card title={`Remediation Strategies (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Strategy</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{r.strategy}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
