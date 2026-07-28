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

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'all-uploads', label: 'All Uploads' },
  { id: 'by-disease', label: 'By Disease' },
  { id: 'by-department', label: 'By Department' },
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

const DISEASE_COLORS = {
  epilepsy: '#3b82f6', parkinsons: '#8b5cf6', sleep_disorder: '#06b6d4',
  depression: '#f59e0b', alzheimers: '#ef4444', unknown: '#94a3b8'
}

const FORMAT_COLORS = {
  '.edf': '#3b82f6', '.bdf': '#8b5cf6', '.fif': '#10b981', '.csv': '#f59e0b', '.set': '#ec4899'
}

const th = { textAlign: 'left', padding: '8px 10px', fontSize: 11, color: '#64748b', fontWeight: 600 }
const td = { padding: '8px 10px' }

export default function UploadsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/uploads/overview`),
      axios.get(`${API_URL}/api/uploads/breakdown`),
      axios.get(`${API_URL}/api/uploads/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading File Uploads data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>File Uploads Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        EEG recording upload analytics &mdash; {overview?.total_uploads} uploads,
        {' '}{overview?.total_patients} patients, {overview?.total_diseases} diseases, {overview?.total_formats} formats
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
      {tab === 'all-uploads' && breakdown && <AllUploadsTab data={breakdown} filter={filter} setFilter={setFilter} />}
      {tab === 'by-disease' && breakdown && <ByDiseaseTab data={breakdown} />}
      {tab === 'by-department' && breakdown && <ByDepartmentTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
      <Card title="Total Uploads"><KPI label="files" value={data.total_uploads} /></Card>
      <Card title="Patients"><KPI label="unique patients" value={data.total_patients} /></Card>
      <Card title="Diseases"><KPI label="disease types" value={data.total_diseases} color="#8b5cf6" /></Card>
      <Card title="Departments"><KPI label="departments" value={data.total_departments} color="#06b6d4" /></Card>
      <Card title="File Formats"><KPI label="format types" value={data.total_formats} color="#10b981" /></Card>

      <Card title="Uploads by Disease" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={data.disease_distribution} dataKey="count" nameKey="disease" cx="50%" cy="50%" outerRadius={80}
            label={({ disease, count }) => `${disease} (${count})`}>
            {data.disease_distribution.map((e, i) => <Cell key={i} fill={DISEASE_COLORS[e.disease] || COLORS[i % COLORS.length]} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Uploads by File Format" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={data.format_distribution} dataKey="count" nameKey="format" cx="50%" cy="50%" outerRadius={80}
            label={({ format, count }) => `${format} (${count})`}>
            {data.format_distribution.map((e, i) => <Cell key={i} fill={FORMAT_COLORS[e.format] || COLORS[i % COLORS.length]} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Uploads by Department" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.department_distribution}><CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="department" tick={{ fontSize: 11 }} /><YAxis /><Tooltip />
            <Bar dataKey="count" fill="#06b6d4" name="Uploads" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Upload Trend" span={3}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.monthly_trend}><CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} /><YAxis /><Tooltip />
            <Line type="monotone" dataKey="uploads" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Uploads" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Top Uploaders (by patient)" span={2}>
        <ResponsiveContainer width="100%" height={Math.max(200, (data.top_uploaders?.length || 0) * 28)}>
          <BarChart data={data.top_uploaders} layout="vertical"><CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" /><YAxis type="category" dataKey="patient_id" width={70} tick={{ fontSize: 11 }} /><Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" name="Uploads" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function AllUploadsTab({ data, filter, setFilter }) {
  const lc = filter.toLowerCase()
  const filtered = data.all_uploads.filter(u =>
    !lc || u.patient_id.toLowerCase().includes(lc) || u.file_name.toLowerCase().includes(lc) ||
    u.disease.toLowerCase().includes(lc) || u.department.toLowerCase().includes(lc)
  )

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`All Uploads (${filtered.length})`} span={1}>
        <input
          type="text" placeholder="Filter by patient, file, disease, department..."
          value={filter} onChange={e => setFilter(e.target.value)}
          style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13, marginBottom: 12 }}
        />
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
              <th style={th}>ID</th><th style={th}>Patient</th><th style={th}>File Name</th>
              <th style={th}>Disease</th><th style={th}>Department</th><th style={th}>Format</th><th style={th}>Date</th>
            </tr></thead>
            <tbody>{filtered.map((u, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}>{u.id}</td>
                <td style={td}>{u.patient_id}</td>
                <td style={{ ...td, fontFamily: 'monospace', fontSize: 11 }}>{u.file_name}</td>
                <td style={td}><Badge text={u.disease} color={DISEASE_COLORS[u.disease]} /></td>
                <td style={td}><Badge text={u.department} color="#06b6d4" /></td>
                <td style={td}><Badge text={u.format} color={FORMAT_COLORS[u.format]} /></td>
                <td style={td}>{u.created_at}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function ByDiseaseTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Disease Summary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr style={{ background: '#f8fafc' }}>
            <th style={th}>Disease</th><th style={th}>Uploads</th><th style={th}>Patients</th><th style={th}>Top Format</th>
          </tr></thead>
          <tbody>{data.disease_summary.map((d, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={td}><Badge text={d.disease} color={DISEASE_COLORS[d.disease]} /></td>
              <td style={td}><strong>{d.count}</strong></td>
              <td style={td}>{d.patients}</td>
              <td style={td}><Badge text={`.${d.top_format}`} color={FORMAT_COLORS[`.${d.top_format}`]} /></td>
            </tr>
          ))}</tbody>
        </table>
      </Card>

      <Card title="Disease x Format Cross-Tab">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>Disease</th>
              {data.formats.map(f => <th key={f} style={{ ...th, textAlign: 'center' }}>{f}</th>)}
            </tr></thead>
            <tbody>{data.disease_format_cross.map((row, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}><Badge text={row.disease} color={DISEASE_COLORS[row.disease]} /></td>
                {data.formats.map(f => (
                  <td key={f} style={{ ...td, textAlign: 'center', fontWeight: row[f] > 0 ? 600 : 400, color: row[f] > 0 ? '#1e293b' : '#cbd5e1' }}>
                    {row[f]}
                  </td>
                ))}
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      <Card title="Uploads by Disease">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data.disease_summary}><CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="disease" tick={{ fontSize: 11 }} /><YAxis /><Tooltip />
            <Bar dataKey="count" name="Uploads" radius={[4, 4, 0, 0]}>
              {data.disease_summary.map((e, i) => <Cell key={i} fill={DISEASE_COLORS[e.disease] || COLORS[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ByDepartmentTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Department Summary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead><tr style={{ background: '#f8fafc' }}>
            <th style={th}>Department</th><th style={th}>Uploads</th><th style={th}>Patients</th><th style={th}>Top Disease</th>
          </tr></thead>
          <tbody>{data.department_summary.map((d, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={td}><Badge text={d.department} color="#06b6d4" /></td>
              <td style={td}><strong>{d.count}</strong></td>
              <td style={td}>{d.patients}</td>
              <td style={td}><Badge text={d.top_disease} color={DISEASE_COLORS[d.top_disease]} /></td>
            </tr>
          ))}</tbody>
        </table>
      </Card>

      <Card title="Uploads by Department">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data.department_summary} layout="vertical"><CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" /><YAxis type="category" dataKey="department" width={90} tick={{ fontSize: 11 }} /><Tooltip />
            <Bar dataKey="count" fill="#06b6d4" name="Uploads" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  const sections = [
    { title: 'Field Definitions', items: data.fields },
    { title: 'File Formats', items: data.file_formats },
    { title: 'Departments', items: data.departments },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map((sec, i) => (
        <Card key={i} title={sec.title} span={sec.items && Object.keys(sec.items).length > 5 ? 2 : 1}>
          {sec.items && Object.entries(sec.items).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 12, color: '#334155' }}>{k}: </span>
              <span style={{ fontSize: 12, color: '#64748b' }}>{v}</span>
            </div>
          ))}
        </Card>
      ))}

      {data.clinical_notes && (
        <Card title="Clinical Notes" span={2}>
          {data.clinical_notes.map((note, i) => (
            <div key={i} style={{ marginBottom: 8, fontSize: 12, color: '#64748b', paddingLeft: 12, borderLeft: '3px solid #e2e8f0' }}>
              {note}
            </div>
          ))}
        </Card>
      )}
    </div>
  )
}
