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

const STATUS_COLORS = { completed: '#10b981', in_progress: '#f59e0b', abandoned: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'sessions', label: 'Session Log' },
  { id: 'instruments', label: 'Instruments' },
  { id: 'patients', label: 'Patients' },
  { id: 'definitions', label: 'Definitions' },
]

export default function GuidedAssessmentDashboard() {
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
      axios.get(`${API_URL}/api/guided-assessment/overview`),
      axios.get(`${API_URL}/api/guided-assessment/breakdown`),
      axios.get(`${API_URL}/api/guided-assessment/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Guided Assessment data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No guided assessment data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Guided Assessment Flow Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Item-by-item clinical assessments via conversational AI and voice channels — completion tracking, instrument analytics, patient history
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
      {tab === 'sessions' && <SessionLogTab breakdown={breakdown} />}
      {tab === 'instruments' && <InstrumentsTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = overview.kpis || []
  const statusDist = overview.status_distribution || []
  const instDist = overview.instrument_distribution || []
  const channelDist = overview.channel_distribution || []
  const dailyTrend = overview.daily_trend || []
  const durHist = overview.duration_histogram || []
  const scoreDist = overview.score_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color || COLORS[i % COLORS.length]} /></Card>
      ))}

      {/* Status Distribution Pie */}
      <Card title="Session Status" span={2}>
        {statusDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={statusDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100}
                label={({ name, value }) => `${name} (${value})`}>
                {statusDist.map((entry, i) => (
                  <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No status data</div>}
      </Card>

      {/* Instrument Distribution Bar */}
      <Card title="Instrument Usage" span={2}>
        {instDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={instDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {instDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No instrument data</div>}
      </Card>

      {/* Channel Distribution Pie */}
      <Card title="Channel Breakdown" span={2}>
        {channelDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={channelDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
                label={({ name, value }) => `${name} (${value})`}>
                {channelDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No channel data</div>}
      </Card>

      {/* Daily Trend Line */}
      <Card title="Daily Session Trend" span={2}>
        {dailyTrend.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={dailyTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="started" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Started" />
              <Line type="monotone" dataKey="completed" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Completed" />
              <Line type="monotone" dataKey="abandoned" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Abandoned" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily trend data</div>}
      </Card>

      {/* Duration Histogram */}
      <Card title="Session Duration" span={2}>
        {durHist.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={durHist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No duration data</div>}
      </Card>

      {/* Score Distribution */}
      <Card title="Score Distribution (completed)" span={2}>
        {scoreDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={scoreDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No score data</div>}
      </Card>
    </div>
  )
}

function SessionLogTab({ breakdown }) {
  const sessions = breakdown?.session_log || []
  const active = breakdown?.active_sessions || []
  const [search, setSearch] = useState('')

  const filtered = search
    ? sessions.filter(s =>
        (s.patient_id || '').toLowerCase().includes(search.toLowerCase()) ||
        (s.patient_name || '').toLowerCase().includes(search.toLowerCase()) ||
        (s.instrument || '').toLowerCase().includes(search.toLowerCase()) ||
        (s.status || '').toLowerCase().includes(search.toLowerCase())
      )
    : sessions

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Active Sessions */}
      {active.length > 0 && (
        <Card title={`Active Sessions (${active.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef3c7' }}>
                  {['Patient', 'Instrument', 'Progress', 'Channel', 'Started'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#92400e' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {active.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}>{s.patient_name || s.patient_id}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.instrument}</td>
                    <td style={{ padding: '6px 10px' }}>{s.current_item}/{s.total_items} ({s.progress})</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={s.channel} color="#6366f1" /></td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{s.started_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Full Session Log */}
      <Card title={`Session Log (${filtered.length})`}>
        <div style={{ marginBottom: 12 }}>
          <input
            type="text"
            placeholder="Search by patient, instrument, status..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0',
              fontSize: 13, outline: 'none', boxSizing: 'border-box'
            }}
          />
        </div>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Instrument', 'Status', 'Progress', 'Score', 'Interpretation', 'Channel', 'Duration', 'Started'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px' }}>{s.patient_name || s.patient_id}</td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.instrument}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={s.status} color={STATUS_COLORS[s.status] || '#6b7280'} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>{s.current_item}/{s.total_items}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{s.score}/{s.max_score}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{s.interpretation}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={s.channel} color="#6366f1" /></td>
                  <td style={{ padding: '6px 10px', fontSize: 11 }}>{s.duration}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{s.started_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function InstrumentsTab({ breakdown }) {
  const instruments = breakdown?.instrument_summary || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Instrument Summary (${instruments.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Instrument', 'Total', 'Completed', 'In Progress', 'Abandoned', 'Completion Rate', 'Avg Duration', 'Avg Score'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Instrument' ? 'left' : 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {instruments.map((inst, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{inst.instrument}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{inst.total}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#10b981' }}>{inst.completed}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#f59e0b' }}>{inst.in_progress}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#ef4444' }}>{inst.abandoned}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{inst.completion_rate}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{inst.avg_duration}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{inst.avg_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Instrument bar comparison */}
      <Card title="Sessions by Instrument" span={1}>
        {instruments.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={instruments}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="instrument" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="completed" fill="#10b981" name="Completed" radius={[4, 4, 0, 0]} />
              <Bar dataKey="in_progress" fill="#f59e0b" name="In Progress" radius={[4, 4, 0, 0]} />
              <Bar dataKey="abandoned" fill="#ef4444" name="Abandoned" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No instrument data</div>}
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const patients = breakdown?.patient_summary || []
  const [search, setSearch] = useState('')

  const filtered = search
    ? patients.filter(p =>
        (p.patient_id || '').toLowerCase().includes(search.toLowerCase()) ||
        (p.patient_name || '').toLowerCase().includes(search.toLowerCase()) ||
        (p.instruments || '').toLowerCase().includes(search.toLowerCase())
      )
    : patients

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Patient Summary (${filtered.length})`}>
        <div style={{ marginBottom: 12 }}>
          <input
            type="text"
            placeholder="Search patients..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0',
              fontSize: 13, outline: 'none', boxSizing: 'border-box'
            }}
          />
        </div>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient ID', 'Name', 'Total Sessions', 'Completed', 'Instruments Used', 'Instruments'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Patient ID' || h === 'Name' || h === 'Instruments' ? 'left' : 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{p.patient_name}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.total_sessions}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#10b981', fontWeight: 600 }}>{p.completed}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.instruments_used}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11 }}>{p.instruments}</td>
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
  const instruments = definitions.instruments || []
  const qualityMetrics = definitions.quality_metrics || []
  const compliance = definitions.compliance || []
  const remediation = definitions.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Guided Assessment Concepts (${concepts.length})`}>
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

      <Card title={`Supported Instruments (${instruments.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Instrument</th>
              <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Items</th>
              <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Max Score</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {instruments.map((inst, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600 }}>{inst.name}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>{inst.items}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>{inst.max_score}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{inst.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

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

      <Card title={`Compliance (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Requirement</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title={`Remediation (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Issue</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{r.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
