import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, RadarChart, Radar, PolarGrid,
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

export default function CognitiveProfileDashboard() {
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
      axios.get(`${API_URL}/api/cognitive-profile/overview`),
      axios.get(`${API_URL}/api/cognitive-profile/breakdown`),
      axios.get(`${API_URL}/api/cognitive-profile/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Cognitive Profile data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No cognitive profile data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Cognitive Profile Summary</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Neuropsychological assessment, cognitive domain scoring, impairment detection (MoCA, MMSE, WAIS, Digit Span, PHQ-9, GAD-7, QOLIE-31, NDDIE)
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
  const domainSummary = overview.domain_summary || []
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
      <Card title="Impairment Severity Distribution" span={2}>
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

      {/* Domain Summary Radar */}
      <Card title="Cognitive Domain Profile (Mean Scores)" span={2}>
        {domainSummary.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={domainSummary}>
              <PolarGrid />
              <PolarAngleAxis dataKey="domain" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Radar name="Mean Score" dataKey="mean_score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
              <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
            </RadarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No domain data</div>}
      </Card>

      {/* Per-instrument Mean Score Bar */}
      <Card title="Mean Normalized Score by Instrument" span={2}>
        {instScores.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={instScores} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="label" width={200} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Bar dataKey="mean_score" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                {instScores.map((entry, i) => (
                  <Cell key={i} fill={entry.mean_score >= 0.8 ? '#10b981' : entry.mean_score >= 0.6 ? '#f59e0b' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No score data</div>}
      </Card>

      {/* Daily Activity Line */}
      <Card title="Daily Assessment Activity" span={4}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No activity data</div>}
      </Card>
    </div>
  )
}

function AssessmentsTab({ breakdown }) {
  if (!breakdown || !breakdown.available) return <div style={{ color: '#94a3b8' }}>No data</div>
  const stats = breakdown.instrument_stats || []
  const inventory = breakdown.assessment_inventory || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Instrument Stats Table */}
      <Card title={`Instrument Statistics (${stats.length} instruments)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: 8 }}>Instrument</th>
                <th style={{ padding: 8 }}>Domain</th>
                <th style={{ padding: 8 }}>N</th>
                <th style={{ padding: 8 }}>Mean Score</th>
                <th style={{ padding: 8 }}>Mean Raw</th>
                <th style={{ padding: 8 }}>Normal</th>
                <th style={{ padding: 8 }}>Mild</th>
                <th style={{ padding: 8 }}>Moderate</th>
                <th style={{ padding: 8 }}>Severe</th>
                <th style={{ padding: 8 }}>Alerts</th>
              </tr>
            </thead>
            <tbody>
              {stats.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{s.label}</td>
                  <td style={{ padding: 8 }}><Badge text={s.domain} color="#6366f1" /></td>
                  <td style={{ padding: 8 }}>{s.count}</td>
                  <td style={{ padding: 8 }}>{(s.mean_score * 100).toFixed(1)}%</td>
                  <td style={{ padding: 8 }}>{s.mean_raw?.toFixed(1)}</td>
                  <td style={{ padding: 8, color: '#10b981' }}>{s.normal}</td>
                  <td style={{ padding: 8, color: '#f59e0b' }}>{s.mild}</td>
                  <td style={{ padding: 8, color: '#f97316' }}>{s.moderate}</td>
                  <td style={{ padding: 8, color: '#ef4444' }}>{s.severe}</td>
                  <td style={{ padding: 8 }}>{s.alerts > 0 ? <Badge text={`${s.alerts}`} color="#ef4444" /> : '0'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Full Assessment Inventory */}
      <Card title={`Assessment Inventory (${inventory.length} records)`}>
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: 6 }}>Patient</th>
                <th style={{ padding: 6 }}>Instrument</th>
                <th style={{ padding: 6 }}>Domain</th>
                <th style={{ padding: 6 }}>Score</th>
                <th style={{ padding: 6 }}>Max</th>
                <th style={{ padding: 6 }}>%</th>
                <th style={{ padding: 6 }}>Level</th>
                <th style={{ padding: 6 }}>Interpretation</th>
                <th style={{ padding: 6 }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {inventory.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontWeight: 600 }}>{a.patient_id}</td>
                  <td style={{ padding: 6 }}>{a.instrument_label}</td>
                  <td style={{ padding: 6 }}><Badge text={a.domain} color="#6366f1" /></td>
                  <td style={{ padding: 6 }}>{a.score}</td>
                  <td style={{ padding: 6 }}>{a.max_score}</td>
                  <td style={{ padding: 6 }}>{a.pct != null ? `${a.pct}%` : '-'}</td>
                  <td style={{ padding: 6 }}>
                    <Badge text={a.normalized_level || a.level || 'unknown'}
                      color={SEVERITY_COLORS[a.normalized_level] || '#6b7280'} />
                  </td>
                  <td style={{ padding: 6, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {a.interpretation || '-'}
                  </td>
                  <td style={{ padding: 6, whiteSpace: 'nowrap' }}>{(a.created_at || '').slice(0, 10)}</td>
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
  if (!breakdown || !breakdown.available) return <div style={{ color: '#94a3b8' }}>No data</div>
  const profiles = breakdown.patient_profiles || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {profiles.map((p, i) => (
        <Card key={i} title={`${p.patient_id} — ${p.name || 'Unknown'}`}>
          <div style={{ display: 'flex', gap: 16, marginBottom: 10, fontSize: 12, color: '#64748b' }}>
            {p.age && <span>Age: {p.age}</span>}
            {p.gender && <span>Gender: {p.gender}</span>}
            {p.disease && <span>Dx: {p.disease}</span>}
          </div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
            <Badge text={`${p.n_assessments} assessments`} color="#3b82f6" />
            <Badge text={p.worst_level} color={SEVERITY_COLORS[p.worst_level] || '#6b7280'} />
            {p.impaired && <Badge text="Cognitive Impairment" color="#ef4444" />}
            {p.alert_count > 0 && <Badge text={`${p.alert_count} alerts`} color="#ef4444" />}
          </div>
          <div style={{ marginBottom: 8, fontSize: 12 }}>
            <strong>Mean Score:</strong>{' '}
            <span style={{ color: p.mean_score >= 0.8 ? '#10b981' : p.mean_score >= 0.6 ? '#f59e0b' : '#ef4444' }}>
              {(p.mean_score * 100).toFixed(1)}%
            </span>
          </div>
          {/* Domain scores */}
          {p.domain_scores && p.domain_scores.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Domain Scores:</div>
              {p.domain_scores.map((d, j) => (
                <div key={j} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, padding: '2px 0' }}>
                  <span style={{ color: '#64748b' }}>{d.domain}</span>
                  <span style={{
                    fontWeight: 600,
                    color: d.mean_score >= 0.8 ? '#10b981' : d.mean_score >= 0.6 ? '#f59e0b' : '#ef4444'
                  }}>{(d.mean_score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          )}
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {(p.instrument_labels || []).map((il, j) => (
              <Badge key={j} text={il} color="#8b5cf6" />
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}

function AlertsTab({ breakdown }) {
  if (!breakdown || !breakdown.available) return <div style={{ color: '#94a3b8' }}>No data</div>
  const alerts = breakdown.clinical_alerts || []

  if (alerts.length === 0) return (
    <Card title="Clinical Alerts">
      <div style={{ color: '#10b981', fontSize: 14, padding: 20, textAlign: 'center' }}>
        No clinical alerts flagged
      </div>
    </Card>
  )

  return (
    <Card title={`Clinical Alerts (${alerts.length})`}>
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8 }}>Patient</th>
              <th style={{ padding: 8 }}>Instrument</th>
              <th style={{ padding: 8 }}>Domain</th>
              <th style={{ padding: 8 }}>Level</th>
              <th style={{ padding: 8 }}>Alert</th>
              <th style={{ padding: 8 }}>Score</th>
              <th style={{ padding: 8 }}>Date</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((a, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{a.patient_id}</td>
                <td style={{ padding: 8 }}>{a.instrument}</td>
                <td style={{ padding: 8 }}><Badge text={a.domain} color="#6366f1" /></td>
                <td style={{ padding: 8 }}><Badge text={a.level || 'unknown'} color={SEVERITY_COLORS[a.level] || '#6b7280'} /></td>
                <td style={{ padding: 8, color: '#ef4444', fontWeight: 600, maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {a.alert}
                </td>
                <td style={{ padding: 8 }}>{a.score}/{a.max_score}</td>
                <td style={{ padding: 8, whiteSpace: 'nowrap' }}>{(a.created_at || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function PipelineTab({ breakdown }) {
  if (!breakdown || !breakdown.available) return <div style={{ color: '#94a3b8' }}>No data</div>
  const events = breakdown.pipeline_events || []

  return (
    <Card title={`Pipeline Event Log (${events.length} events)`}>
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 6 }}>ID</th>
              <th style={{ padding: 6 }}>Component</th>
              <th style={{ padding: 6 }}>Action</th>
              <th style={{ padding: 6 }}>Actor</th>
              <th style={{ padding: 6 }}>Detail</th>
              <th style={{ padding: 6 }}>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{ev.id}</td>
                <td style={{ padding: 6 }}><Badge text={ev.component || '-'} color="#3b82f6" /></td>
                <td style={{ padding: 6 }}>{ev.action}</td>
                <td style={{ padding: 6 }}>{ev.actor || '-'}</td>
                <td style={{ padding: 6, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {ev.detail || '-'}
                </td>
                <td style={{ padding: 6, whiteSpace: 'nowrap' }}>{ev.ts_local || ev.ts_utc || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <div style={{ color: '#94a3b8' }}>No definitions</div>
  const concepts = definitions.concepts || []
  const metrics = definitions.quality_metrics || []
  const instruments = definitions.assessment_instruments || []
  const compliance = definitions.compliance || []
  const remediation = definitions.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Concepts */}
      <Card title={`Cognitive Assessment Concepts (${concepts.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 200 }}>Concept</th>
              <th style={{ padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, verticalAlign: 'top' }}>{c.name}</td>
                <td style={{ padding: 8, color: '#475569', lineHeight: 1.5 }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Assessment Instruments */}
      <Card title={`Assessment Instruments (${instruments.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8 }}>Instrument</th>
              <th style={{ padding: 8 }}>Full Name</th>
              <th style={{ padding: 8 }}>Domain</th>
            </tr>
          </thead>
          <tbody>
            {instruments.map((inst, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{inst.instrument}</td>
                <td style={{ padding: 8 }}>{inst.label}</td>
                <td style={{ padding: 8 }}><Badge text={inst.domain} color="#6366f1" /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Quality Metrics */}
      <Card title={`Quality Metrics (${metrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 200 }}>Metric</th>
              <th style={{ padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, verticalAlign: 'top' }}>{m.name}</td>
                <td style={{ padding: 8, color: '#475569', lineHeight: 1.5 }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Compliance */}
      <Card title={`Compliance References (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 180 }}>Reference</th>
              <th style={{ padding: 8 }}>Note</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, verticalAlign: 'top' }}>{c.ref}</td>
                <td style={{ padding: 8, color: '#475569', lineHeight: 1.5 }}>{c.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Remediation */}
      <Card title={`Remediation Strategies (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 220 }}>Strategy</th>
              <th style={{ padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, verticalAlign: 'top' }}>{r.strategy}</td>
                <td style={{ padding: 8, color: '#475569', lineHeight: 1.5 }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
