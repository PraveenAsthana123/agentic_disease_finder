import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patient-detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function RecordingConditionsDashboard() {
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
      axios.get(`${API_URL}/api/recording-conditions/overview`),
      axios.get(`${API_URL}/api/recording-conditions/breakdown`),
      axios.get(`${API_URL}/api/recording-conditions/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Recording Conditions data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Recording Conditions Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          EEG recording conditions — activation procedures, patient state, cooperation levels, protocol completeness
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'patient-detail' && breakdown && <PatientDetailTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

/* -- Overview Tab --------------------------------------------------------- */
function OverviewTab({ data }) {
  const rates = data.activation_rates || {}
  const stateDist = data.patient_state_distribution || {}
  const coopDist = data.cooperation_distribution || {}
  const quality = data.quality_summary || {}

  const activationData = [
    { name: 'Eyes Open', value: rates.eyes_open_pct || 0 },
    { name: 'Hyperventilation', value: rates.hyperventilation_pct || 0 },
    { name: 'Photic Stimulation', value: rates.photic_stimulation_pct || 0 },
    { name: 'Sleep Recorded', value: rates.sleep_recorded_pct || 0 },
  ]

  const stateData = Object.entries(stateDist).map(([key, val]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1), value: val,
  }))

  const coopData = Object.entries(coopDist).map(([key, val]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1), value: val,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI row */}
      <Card>
        <KPI label="Total Recordings" value={data.total_recordings} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Protocol Completeness" value={`${data.protocol_completeness ?? 0}%`}
          color={data.protocol_completeness >= 80 ? '#10b981' : '#f59e0b'} />
      </Card>
      <Card>
        <KPI label="Excellent + Good" value={`${quality.excellent_good_pct ?? 0}%`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Fair + Poor" value={`${quality.fair_poor_pct ?? 0}%`}
          color={quality.fair_poor_pct > 20 ? '#ef4444' : '#f59e0b'} />
      </Card>

      {/* Activation Procedure Rates bar chart */}
      <Card title="Activation Procedure Rates (%)" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={activationData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#64748b' }} />
            <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
              {activationData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Patient State Distribution pie chart */}
      <Card title="Patient State Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={stateData} dataKey="value" nameKey="name" cx="50%" cy="50%"
              outerRadius={80} innerRadius={40} paddingAngle={3} label={({ name, percent }) =>
                `${name} ${(percent * 100).toFixed(0)}%`
              } labelLine={{ stroke: '#94a3b8' }}>
              {stateData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Cooperation Level Distribution pie chart */}
      <Card title="Cooperation Level Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={coopData} dataKey="value" nameKey="name" cx="50%" cy="50%"
              outerRadius={80} innerRadius={40} paddingAngle={3} label={({ name, percent }) =>
                `${name} ${(percent * 100).toFixed(0)}%`
              } labelLine={{ stroke: '#94a3b8' }}>
              {coopData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* -- Patient Detail Tab --------------------------------------------------- */
function PatientDetailTab({ data }) {
  const patients = data.patients || []

  const check = (val) => (
    <span style={{ color: val ? '#10b981' : '#ef4444', fontWeight: 700, fontSize: 16 }}>
      {val ? '\u2713' : '\u2717'}
    </span>
  )

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Patient Recording Details" span={1}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient ID', 'Eyes Open', 'Hyperventilation', 'Photic Stimulation',
                  'Sleep Recorded', 'Patient State', 'Cooperation', 'Activations', 'Protocol Complete'
                ].map((h, i) => (
                  <th key={i} style={{
                    padding: '8px 10px', textAlign: 'left',
                    borderBottom: '2px solid #e2e8f0', whiteSpace: 'nowrap',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{
                  background: p.protocol_complete ? '#f0fdf410' : undefined,
                  backgroundColor: p.protocol_complete ? 'rgba(240,253,244,0.4)' : undefined,
                }}>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#1e293b' }}>
                    {p.patient_id}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                    {check(p.eyes_open)}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                    {check(p.hyperventilation)}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                    {check(p.photic_stimulation)}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                    {check(p.sleep_recorded)}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', color: '#475569', textTransform: 'capitalize' }}>
                    {p.patient_state}
                  </td>
                  <td style={{
                    padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textTransform: 'capitalize',
                    color: p.cooperation === 'poor' ? '#ef4444' : '#475569',
                    fontWeight: p.cooperation === 'poor' ? 700 : 400,
                  }}>
                    {p.cooperation}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center', color: '#475569' }}>
                    {p.activations_completed}/4
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                    {check(p.protocol_complete)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* -- Definitions Tab ------------------------------------------------------ */
function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={data.title || 'Definitions'}>
        {data.description && (
          <p style={{ fontSize: 13, color: '#64748b', marginTop: 0, marginBottom: 16 }}>
            {data.description}
          </p>
        )}
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left',
                borderBottom: '2px solid #e2e8f0', width: 200 }}>Term</th>
              <th style={{ padding: '8px 10px', textAlign: 'left',
                borderBottom: '2px solid #e2e8f0' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {(data.terms || []).map((row, i) => (
              <tr key={i}>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                  fontWeight: 600, color: '#1e293b' }}>
                  {row.term}
                </td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                  color: '#475569' }}>
                  {row.definition}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
