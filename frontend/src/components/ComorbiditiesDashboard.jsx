import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function SeverityBadge({ severity }) {
  const s = String(severity || '').toLowerCase()
  const color = s === 'minimal' ? '#10b981' : s === 'mild' ? '#3b82f6' : s === 'moderate' ? '#f59e0b' : s === 'severe' ? '#ef4444' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s.replace(/_/g, ' ') || '--'}</span>
  )
}

function TreatmentBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'stable' ? '#10b981' : s === 'under_treatment' ? '#3b82f6' : s === 'untreated' ? '#f59e0b' : s === 'treatment_resistant' ? '#ef4444' : s === 'none' ? '#94a3b8' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s.replace(/_/g, ' ') || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'All Patients' },
  { id: 'conditions', label: 'By Condition' },
  { id: 'severity', label: 'By Severity' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ComorbiditiesDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/comorbidities/overview`),
      axios.get(`${API_URL}/api/comorbidities/breakdown`),
      axios.get(`${API_URL}/api/comorbidities/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading comorbidities data…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Comorbidities Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Psychiatric comorbidity screening analytics — {fmt(overview?.total_patients)} patients, {fmt(overview?.comorbidity_rate)}% comorbidity rate, {fmt(overview?.avg_behavioral_risk_score)} avg risk score
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'patients' && <PatientsTab patients={breakdown?.patients || []} />}
      {tab === 'conditions' && <ConditionsTab data={breakdown} overview={overview} />}
      {tab === 'severity' && <SeverityTab data={overview} breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const condData = (data.condition_distribution || []).slice(0, 8).map(d => ({ name: d.condition, value: d.count }))
  const sevData = (data.severity_distribution || []).map(d => ({ name: d.severity, value: d.count }))
  const treatData = (data.treatment_distribution || []).map(d => ({ name: d.status.replace(/_/g, ' '), value: d.count }))
  const impactData = (data.impact_distribution || []).map(d => ({ name: d.impact, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
          <KPI label="Total Patients" value={data.total_patients} color="#3b82f6" />
          <KPI label="With Comorbidities" value={data.total_with_comorbidities} sub={`${data.comorbidity_rate}%`} color="#ef4444" />
          <KPI label="Avg Conditions" value={data.avg_comorbidity_count} sub={`max ${data.max_comorbidity_count}`} color="#8b5cf6" />
          <KPI label="Avg Risk Score" value={data.avg_behavioral_risk_score} sub="0-100 scale" color="#f59e0b" />
          <KPI label="Screening Rate" value={`${data.screening_rate}%`} sub={`${data.screened_count} screened`} color="#10b981" />
        </div>
      </Card>

      <Card title="Condition Prevalence (Top 8)">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={condData} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={100} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Patients" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Risk Severity Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
              {sevData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Treatment Status">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={treatData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
              {treatData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Functional Impact">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={impactData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#f59e0b" name="Patients" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Comorbidity Count Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.count_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 12 }} label={{ value: 'Conditions', position: 'insideBottom', offset: -2 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" name="Patients" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Screening Instruments Used">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.instrument_distribution || []} layout="vertical" margin={{ left: 50 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="instrument" type="category" tick={{ fontSize: 11 }} width={50} />
            <Tooltip />
            <Bar dataKey="count" fill="#06b6d4" name="Uses" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Screening Trend" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="screened" fill="#3b82f6" name="Screened" radius={[4, 4, 0, 0]} />
            <Bar dataKey="with_comorbidity" fill="#ef4444" name="With Comorbidity" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientsTab({ patients }) {
  const [sort, setSort] = useState('behavioral_risk_score')
  const [dir, setDir] = useState(-1)
  const [filter, setFilter] = useState('')

  const toggle = col => {
    if (sort === col) setDir(-dir)
    else { setSort(col); setDir(-1) }
  }

  const filtered = patients.filter(p =>
    !filter || p.patient_id?.toLowerCase().includes(filter.toLowerCase()) ||
    p.comorbidities?.toLowerCase().includes(filter.toLowerCase()) ||
    p.risk_severity?.toLowerCase().includes(filter.toLowerCase())
  )
  const sorted = [...filtered].sort((a, b) => {
    const av = a[sort], bv = b[sort]
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir
    return String(av || '').localeCompare(String(bv || '')) * dir
  })

  const hdr = (label, col) => (
    <th onClick={() => toggle(col)} style={{ padding: '8px 10px', cursor: 'pointer', fontSize: 12, color: '#475569', fontWeight: 600, textAlign: 'left', whiteSpace: 'nowrap' }}>
      {label} {sort === col ? (dir > 0 ? '▲' : '▼') : ''}
    </th>
  )

  return (
    <Card title={`All Patients (${sorted.length})`}>
      <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Filter by patient, condition, or severity…"
        style={{ width: '100%', maxWidth: 400, padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', marginBottom: 12, fontSize: 13 }} />
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead style={{ background: '#f8fafc' }}>
            <tr>
              {hdr('Patient', 'patient_id')}
              {hdr('Count', 'comorbidity_count')}
              {hdr('Conditions', 'comorbidities')}
              {hdr('Risk Severity', 'risk_severity')}
              {hdr('Risk Score', 'behavioral_risk_score')}
              {hdr('Impact', 'functional_impact')}
              {hdr('Treatment', 'treatment_status')}
              {hdr('Screen Date', 'screening_date')}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.comorbidity_count}</td>
                <td style={{ padding: '6px 10px', maxWidth: 260, fontSize: 12 }}>{p.comorbidities}</td>
                <td style={{ padding: '6px 10px' }}><SeverityBadge severity={p.risk_severity} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(p.behavioral_risk_score)}</td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{(p.functional_impact || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px' }}><TreatmentBadge status={p.treatment_status} /></td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{p.screening_date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function ConditionsTab({ data, overview }) {
  const conditions = data?.by_condition || []
  const condDist = (overview?.condition_distribution || []).map(d => ({ name: d.condition, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
      <Card title="Condition Prevalence" span={2}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={condDist} layout="vertical" margin={{ left: 160 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={160} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Patients" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Condition Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead style={{ background: '#f8fafc' }}>
              <tr>
                <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Condition</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', fontSize: 12, color: '#475569', fontWeight: 600 }}>Patients</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', fontSize: 12, color: '#475569', fontWeight: 600 }}>Avg Risk Score</th>
              </tr>
            </thead>
            <tbody>
              {conditions.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.condition}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{c.patient_count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(c.avg_risk_score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function SeverityTab({ data, breakdown }) {
  const riskBySev = (data?.risk_by_severity || []).map(d => ({ name: d.severity, score: d.avg_score, count: d.count }))
  const bySeverity = breakdown?.by_severity || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
      <Card title="Average Risk Score by Severity">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={riskBySev}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Bar dataKey="score" fill="#ef4444" name="Avg Risk Score" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Patient Count by Severity">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={riskBySev} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={90}
              label={({ name, count }) => `${name}: ${count}`}>
              {riskBySev.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Severity Group Details" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead style={{ background: '#f8fafc' }}>
              <tr>
                <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Severity</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', fontSize: 12, color: '#475569', fontWeight: 600 }}>Patient Count</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Patients</th>
              </tr>
            </thead>
            <tbody>
              {bySeverity.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px' }}><SeverityBadge severity={s.severity} /></td>
                  <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{s.patient_count}</td>
                  <td style={{ padding: '6px 10px', fontSize: 12 }}>{(s.patients || []).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
      <Card title="Field Definitions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead style={{ background: '#f8fafc' }}>
            <tr>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Field</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {(data.fields || []).map((f, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>{f.field}</td>
                <td style={{ padding: '6px 10px' }}>{f.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Comorbid Conditions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead style={{ background: '#f8fafc' }}>
            <tr>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Condition</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {(data.conditions || []).map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.name}</td>
                <td style={{ padding: '6px 10px' }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Screening Instruments">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead style={{ background: '#f8fafc' }}>
            <tr>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Instrument</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', fontWeight: 600 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {(data.instruments || []).map((inst, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, fontFamily: 'monospace' }}>{inst.name}</td>
                <td style={{ padding: '6px 10px' }}>{inst.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Data Source">
        <p style={{ fontSize: 13, color: '#475569', margin: 0 }}>{data.data_source}</p>
      </Card>
    </div>
  )
}
