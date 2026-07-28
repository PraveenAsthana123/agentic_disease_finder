import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
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

function SexBadge({ sex }) {
  const s = String(sex || '').toLowerCase()
  const color = s === 'male' ? '#3b82f6' : s === 'female' ? '#ec4899' : '#8b5cf6'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s || '--'}</span>
  )
}

function InsuranceBadge({ insurance }) {
  const ins = String(insurance || '').toLowerCase()
  const color = ins === 'private' ? '#10b981' : ins === 'medicare' ? '#3b82f6' : ins === 'medicaid' ? '#f59e0b' : ins === 'uninsured' ? '#ef4444' : '#8b5cf6'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{ins || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'All Patients' },
  { id: 'epilepsy-type', label: 'By Epilepsy Type' },
  { id: 'neurologist', label: 'By Neurologist' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PatientDemographicsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/patient-demographics/overview`),
      axios.get(`${API_URL}/api/patient-demographics/breakdown`),
      axios.get(`${API_URL}/api/patient-demographics/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading patient demographics...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Patient Demographics Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Patient demographic analytics — {fmt(overview?.kpis?.total_patients)} patients, avg age {fmt(overview?.kpis?.avg_age)}, avg BMI {fmt(overview?.kpis?.avg_bmi)}
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
      {tab === 'epilepsy-type' && <EpilepsyTypeTab data={breakdown} />}
      {tab === 'neurologist' && <NeurologistTab data={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const k = data.kpis || {}
  const sexData = (data.sex_dist || []).map(d => ({ name: d.sex, value: d.count }))
  const epilepsyData = (data.epilepsy_type_dist || []).map(d => ({ name: d.epilepsy_type, value: d.count }))
  const bmiCatData = (data.bmi_category_dist || []).map(d => ({ name: d.category, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
          <KPI label="Total Patients" value={k.total_patients} color="#3b82f6" />
          <KPI label="Avg Age" value={k.avg_age} sub="years" color="#8b5cf6" />
          <KPI label="Avg BMI" value={k.avg_bmi} sub="kg/m²" color="#06b6d4" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          <KPI label="Interpreter Needed" value={`${k.interpreter_needed_pct}%`} color="#f59e0b" />
          <KPI label="Avg Years w/ Epilepsy" value={k.avg_years_with_epilepsy} sub="years" color="#ef4444" />
          <KPI label="Avg Onset Age" value={k.avg_onset_age} sub="years" color="#ec4899" />
        </div>
      </Card>

      <Card title="Sex Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={sexData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {sexData.map((d, i) => {
                const c = d.name === 'Male' ? '#3b82f6' : d.name === 'Female' ? '#ec4899' : '#8b5cf6'
                return <Cell key={i} fill={c} />
              })}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Epilepsy Type Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={epilepsyData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {epilepsyData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Insurance Type">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.insurance_type_dist || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="insurance_type" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
              {(data.insurance_type_dist || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Age Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.age_histogram || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="age_bin" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]} fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Enrollment Trend" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.enrollment_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="patients" stroke="#3b82f6" name="Patients Enrolled" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="BMI Categories">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={bmiCatData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {bmiCatData.map((d, i) => {
                const c = d.name === 'Normal' ? '#10b981' : d.name === 'Overweight' ? '#f59e0b' : d.name === 'Obese' ? '#ef4444' : '#3b82f6'
                return <Cell key={i} fill={c} />
              })}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Referral Source">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.referral_source_dist || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="referral_source" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
              {(data.referral_source_dist || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="State Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.state_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="state" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]} fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Education Level">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.education_level_dist || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="education_level" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
              {(data.education_level_dist || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Employment Status">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.employment_status_dist || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="employment_status" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
              {(data.employment_status_dist || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientsTab({ patients }) {
  const [sortKey, setSortKey] = useState('patient_id')
  const [sortDir, setSortDir] = useState(1)
  const [filter, setFilter] = useState('')

  const filtered = patients.filter(p => {
    if (!filter) return true
    const f = filter.toLowerCase()
    return (p.patient_id || '').toLowerCase().includes(f) ||
           (p.full_name || '').toLowerCase().includes(f)
  })

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * sortDir
  })

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d * -1)
    else { setSortKey(key); setSortDir(1) }
  }

  const hdr = (label, key) => (
    <th onClick={() => toggleSort(key)} style={{
      padding: '8px 10px', cursor: 'pointer', whiteSpace: 'nowrap', fontSize: 12,
      background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left',
      color: sortKey === key ? '#3b82f6' : '#475569'
    }}>{label} {sortKey === key ? (sortDir > 0 ? '\u25B2' : '\u25BC') : ''}</th>
  )

  return (
    <Card title={`All Patients (${filtered.length})`}>
      <input type="text" placeholder="Filter by patient ID or name..." value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8,
          fontSize: 13, marginBottom: 12, boxSizing: 'border-box' }} />
      <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('Patient ID', 'patient_id')}
              {hdr('Name', 'full_name')}
              {hdr('Age', 'age')}
              <th style={{ padding: '8px 10px', fontSize: 12, background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569' }}>Sex</th>
              {hdr('Epilepsy Type', 'epilepsy_type')}
              <th style={{ padding: '8px 10px', fontSize: 12, background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569' }}>Insurance</th>
              {hdr('BMI', 'bmi')}
              {hdr('Yrs w/ Epilepsy', 'years_with_epilepsy')}
              {hdr('Neurologist', 'primary_neurologist')}
              {hdr('State', 'address_state')}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 100).map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                <td style={{ padding: '6px 10px' }}>{p.full_name}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(p.age)}</td>
                <td style={{ padding: '6px 10px' }}><SexBadge sex={p.sex} /></td>
                <td style={{ padding: '6px 10px' }}>{p.epilepsy_type}</td>
                <td style={{ padding: '6px 10px' }}><InsuranceBadge insurance={p.insurance_type} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(p.bmi)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(p.years_with_epilepsy)}</td>
                <td style={{ padding: '6px 10px' }}>{p.primary_neurologist}</td>
                <td style={{ padding: '6px 10px' }}>{p.address_state}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 100 && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing first 100 of {sorted.length} patients</div>}
    </Card>
  )
}

function EpilepsyTypeTab({ data }) {
  if (!data) return null
  const types = data.by_epilepsy_type || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
      <Card title={`Epilepsy Type Analysis (${types.length} types)`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Epilepsy Type</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Patients</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Age</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Years w/ Epilepsy</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg BMI</th>
              </tr>
            </thead>
            <tbody>
              {types.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{t.epilepsy_type}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(t.patient_count)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(t.avg_age)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(t.avg_years_with_epilepsy)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(t.avg_bmi)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Patients by Epilepsy Type" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={types}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epilepsy_type" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="patient_count" name="Patients" radius={[4, 4, 0, 0]}>
              {types.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function NeurologistTab({ data }) {
  if (!data) return null
  const neuros = data.by_neurologist || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
      <Card title={`Neurologist Panel (${neuros.length} neurologists)`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Neurologist</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Patients</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Age</th>
              </tr>
            </thead>
            <tbody>
              {neuros.map((n, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{n.primary_neurologist}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(n.patient_count)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(n.avg_age)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Patient Count by Neurologist" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={neuros}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="primary_neurologist" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="patient_count" name="Patients" radius={[4, 4, 0, 0]}>
              {neuros.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return <Card>No definitions available.</Card>
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
      <Card title={data.title || 'Definitions'} span={2}>
        {(data.concepts || []).map((c, i) => (
          <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < data.concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.name}</div>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.description}</div>
          </div>
        ))}
      </Card>

      {(data.data_sources || []).length > 0 && (
        <Card title="Data Sources">
          <ul style={{ margin: 0, padding: '0 0 0 20px', fontSize: 13, color: '#64748b' }}>
            {data.data_sources.map((src, i) => <li key={i} style={{ marginBottom: 4 }}>{src}</li>)}
          </ul>
        </Card>
      )}
    </div>
  )
}

const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569' }
const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }
