import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line
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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316']
const RISK_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#10b981', critical: '#dc2626' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
      fontWeight: 600, background: (color || '#94a3b8') + '22', color: color || '#94a3b8'
    }}>{text}</span>
  )
}

export default function WearableReadingsDashboard() {
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
      axios.get(`${API_URL}/api/wearable-readings/overview`),
      axios.get(`${API_URL}/api/wearable-readings/breakdown`),
      axios.get(`${API_URL}/api/wearable-readings/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Wearable Readings data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Wearable Readings Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Wearable health monitoring analytics — {overview?.total_readings} readings,
        {' '}{overview?.total_patients} patients, {overview?.total_devices} devices
      </p>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const activityPie = data.activity_distribution.map((a, i) => ({ ...a, fill: COLORS[i % COLORS.length] }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total Readings"><KPI label="Readings" value={data.total_readings} /></Card>
      <Card title="Total Patients"><KPI label="Patients" value={data.total_patients} color="#3b82f6" /></Card>
      <Card title="Total Devices"><KPI label="Devices" value={data.total_devices} color="#8b5cf6" /></Card>
      <Card title="Avg Heart Rate"><KPI label="bpm" value={data.avg_heart_rate} color="#ef4444" /></Card>

      <Card title="Avg Steps"><KPI label="steps/day" value={data.avg_steps} color="#10b981" /></Card>
      <Card title="Avg Sleep Hours"><KPI label="hours" value={data.avg_sleep_hours} color="#06b6d4" /></Card>
      <Card title="Avg SpO2"><KPI label="%" value={data.avg_spo2} color="#3b82f6" /></Card>
      <Card title="Avg Health Score"><KPI label="score" value={data.avg_health_score} color="#8b5cf6" /></Card>

      <Card title="Seizure Events"><KPI label="total detected" value={data.seizure_events} color="#ef4444" /></Card>
      <Card title="Fall Events"><KPI label="total detected" value={data.fall_events} color="#f59e0b" /></Card>
      <Card title="Activity Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={activityPie} dataKey="count" nameKey="activity" cx="50%" cy="50%" outerRadius={80} label={({ activity, count }) => `${activity} (${count})`}>
            {activityPie.map((e, i) => <Cell key={i} fill={e.fill} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Heart Rate Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.heart_rate_distribution}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="bucket" /><YAxis /><Tooltip />
            <Bar dataKey="count" fill="#ef4444" name="Readings" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Sleep Quality Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.sleep_quality_distribution}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="quality" /><YAxis /><Tooltip />
            <Bar dataKey="count" fill="#06b6d4" name="Readings">{data.sleep_quality_distribution.map((e, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Daily Trend (Heart Rate & Steps)" span={4}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.daily_trend}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="date" tick={{ fontSize: 11 }} /><YAxis yAxisId="left" label={{ value: 'HR (bpm)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} /><YAxis yAxisId="right" orientation="right" label={{ value: 'Steps', angle: 90, position: 'insideRight', style: { fontSize: 11 } }} /><Tooltip />
            <Line yAxisId="left" type="monotone" dataKey="avg_heart_rate" stroke="#ef4444" name="Avg Heart Rate" strokeWidth={2} dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="avg_steps" stroke="#10b981" name="Avg Steps" strokeWidth={2} dot={false} />
            <Legend />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  const riskColor = (risk) => {
    if (risk == null) return '#94a3b8'
    const r = parseFloat(risk)
    if (r >= 0.7) return '#ef4444'
    if (r >= 0.4) return '#f59e0b'
    return '#10b981'
  }
  const riskLabel = (risk) => {
    if (risk == null) return 'N/A'
    const r = parseFloat(risk)
    if (r >= 0.7) return 'High'
    if (r >= 0.4) return 'Medium'
    return 'Low'
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Per-Patient Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>Patient</th><th style={th}>Device</th><th style={th}>Readings</th><th style={th}>Avg HR</th><th style={th}>Avg Steps</th><th style={th}>Avg Sleep (h)</th><th style={th}>Avg SpO2</th><th style={th}>Seizure Events</th><th style={th}>Avg Seizure Risk</th>
            </tr></thead>
            <tbody>{data.per_patient.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}>{p.patient_id}</td>
                <td style={td}>{p.device_id || '—'}</td>
                <td style={td}>{p.readings}</td>
                <td style={td}>{p.avg_hr}</td>
                <td style={td}>{p.avg_steps}</td>
                <td style={td}>{p.avg_sleep}</td>
                <td style={td}>{p.avg_spo2}</td>
                <td style={td}>{p.seizure_events > 0 ? <Badge text={p.seizure_events} color="#ef4444" /> : '0'}</td>
                <td style={td}><Badge text={`${p.avg_seizure_risk} (${riskLabel(p.avg_seizure_risk)})`} color={riskColor(p.avg_seizure_risk)} /></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      {data.high_risk_patients && data.high_risk_patients.length > 0 && (
        <Card title={`High-Risk Patients (${data.high_risk_patients.length})`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead><tr style={{ background: '#fef2f2' }}>
                <th style={th}>Patient</th><th style={th}>Device</th><th style={th}>Avg Seizure Risk</th><th style={th}>Seizure Events</th><th style={th}>Avg HR</th><th style={th}>Avg SpO2</th>
              </tr></thead>
              <tbody>{data.high_risk_patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={td}>{p.patient_id}</td>
                  <td style={td}>{p.device_id || '—'}</td>
                  <td style={td}><Badge text={p.avg_seizure_risk} color="#ef4444" /></td>
                  <td style={td}><Badge text={p.seizure_events} color="#ef4444" /></td>
                  <td style={td}>{p.avg_hr}</td>
                  <td style={td}>{p.avg_spo2}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </Card>
      )}

      {data.seizure_events_table && data.seizure_events_table.length > 0 && (
        <Card title={`Seizure Events (${data.seizure_events_table.length})`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead><tr style={{ background: '#fef2f2' }}>
                <th style={th}>Patient</th><th style={th}>Date</th><th style={th}>Confidence</th><th style={th}>Heart Rate</th>
              </tr></thead>
              <tbody>{data.seizure_events_table.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={td}>{s.patient_id}</td>
                  <td style={td}>{s.date?.slice(0, 10) || s.timestamp?.slice(0, 10) || '—'}</td>
                  <td style={td}><Badge text={s.confidence != null ? s.confidence : '—'} color={s.confidence >= 0.8 ? '#ef4444' : s.confidence >= 0.5 ? '#f59e0b' : '#10b981'} /></td>
                  <td style={td}>{s.heart_rate ?? '—'} bpm</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Recent Readings (Last 20)" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>Patient</th><th style={th}>Date</th><th style={th}>HR</th><th style={th}>Steps</th><th style={th}>Sleep (h)</th><th style={th}>SpO2</th><th style={th}>Stress</th><th style={th}>Activity</th><th style={th}>Seizure Risk</th><th style={th}>Health Score</th>
            </tr></thead>
            <tbody>{data.recent_readings.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}>{r.patient_id}</td>
                <td style={td}>{r.date?.slice(0, 10) || r.timestamp?.slice(0, 10) || '—'}</td>
                <td style={td}>{r.heart_rate ?? '—'}</td>
                <td style={td}>{r.steps ?? '—'}</td>
                <td style={td}>{r.sleep_hours ?? '—'}</td>
                <td style={td}>{r.spo2 ?? '—'}</td>
                <td style={td}>{r.stress_level ?? '—'}</td>
                <td style={td}>{r.activity_type || '—'}</td>
                <td style={td}><Badge text={`${r.seizure_risk ?? '—'} (${riskLabel(r.seizure_risk)})`} color={riskColor(r.seizure_risk)} /></td>
                <td style={td}>{r.health_score ?? '—'}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

const th = { textAlign: 'left', padding: '8px 10px', fontSize: 11, color: '#64748b', fontWeight: 600 }
const td = { padding: '8px 10px' }

function DefinitionsTab({ data }) {
  const sections = [
    { title: 'Clinical Glossary', items: data.glossary },
    { title: 'Field Definitions', items: data.fields },
    { title: 'Clinical Thresholds', items: data.thresholds },
    { title: 'Clinical Notes', items: data.clinical_notes },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map((sec, i) => (
        <Card key={i} title={sec.title} span={sec.items && Object.keys(sec.items).length > 6 ? 2 : 1}>
          {sec.items && Object.entries(sec.items).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 12, color: '#334155' }}>{k}: </span>
              <span style={{ fontSize: 12, color: '#64748b' }}>{v}</span>
            </div>
          ))}
        </Card>
      ))}
    </div>
  )
}
