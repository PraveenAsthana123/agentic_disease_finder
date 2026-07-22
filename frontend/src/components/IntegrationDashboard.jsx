import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

const card = { background: '#fff', borderRadius: 10, padding: 18, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', marginBottom: 16 }
const kpiBox = { textAlign: 'center', padding: '14px 10px', background: '#f8fafc', borderRadius: 8, minWidth: 120 }
const kpiVal = { fontSize: 26, fontWeight: 700 }
const kpiLabel = { fontSize: 11, color: '#64748b', marginTop: 2 }
const tabBtn = (active) => ({
  padding: '7px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
  background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#475569'
})

export default function IntegrationDashboard() {
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
          axios.get(`${API_URL}/api/integration-dashboard/overview`),
          axios.get(`${API_URL}/api/integration-dashboard/breakdown`),
          axios.get(`${API_URL}/api/integration-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load integration data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8, animation: 'spin 1.5s linear infinite' }}>&#9881;</div>
      Loading integration data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Integration data not available.'}
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patient Uploads' },
    { id: 'activity', label: 'Activity & Trends' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 700 }}>Integration Dashboard</h2>
      <p style={{ margin: '0 0 16px', color: '#64748b', fontSize: 13 }}>
        Multi-format data ingest analytics — uploads, formats, pipeline activity
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} style={tabBtn(tab === t.id)} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'patients' && renderPatients()}
      {tab === 'activity' && renderActivity()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const ov = overview
    return (
      <>
        {/* KPI row */}
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 18 }}>
          <div style={kpiBox}><div style={{ ...kpiVal, color: '#3b82f6' }}>{fmt(ov.total_uploads)}</div><div style={kpiLabel}>Total Uploads</div></div>
          <div style={kpiBox}><div style={{ ...kpiVal, color: '#10b981' }}>{fmt(ov.unique_patients)}</div><div style={kpiLabel}>Unique Patients</div></div>
          <div style={kpiBox}><div style={{ ...kpiVal, color: '#f59e0b' }}>{fmt(ov.unique_files)}</div><div style={kpiLabel}>Unique Files</div></div>
          <div style={kpiBox}><div style={{ ...kpiVal, color: '#8b5cf6' }}>{ov.upload_coverage_pct}%</div><div style={kpiLabel}>Patient Coverage</div></div>
          <div style={kpiBox}><div style={{ ...kpiVal, color: '#ef4444' }}>{fmt(ov.ingest_transactions)}</div><div style={kpiLabel}>Ingest Txns</div></div>
        </div>

        {/* Format distribution + Department */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Format Distribution</h4>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.format_distribution} dataKey="count" nameKey="format" cx="50%" cy="50%" outerRadius={80} label={({ format, count }) => `${format}: ${count}`}>
                  {ov.format_distribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div style={card}>
            <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Department Distribution</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.department_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="dept" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Ingest actors + Disease distribution */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Ingest Actors</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={ov.ingest_actors} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="actor" tick={{ fontSize: 12 }} width={100} />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[0,4,4,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={card}>
            <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Disease Distribution</h4>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={ov.disease_distribution} dataKey="count" nameKey="disease" cx="50%" cy="50%" outerRadius={70} label={({ disease, count }) => `${disease}: ${count}`}>
                  {ov.disease_distribution.map((_, i) => <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Date range */}
        <div style={card}>
          <div style={{ display: 'flex', gap: 30, fontSize: 13 }}>
            <span><strong>First Upload:</strong> {ov.first_upload?.slice(0, 19) || '--'}</span>
            <span><strong>Last Upload:</strong> {ov.last_upload?.slice(0, 19) || '--'}</span>
            <span><strong>Total Patients:</strong> {fmt(ov.total_patients)}</span>
          </div>
        </div>
      </>
    )
  }

  function renderPatients() {
    const bd = breakdown
    if (!bd?.available) return <div>No breakdown data available.</div>

    return (
      <>
        {/* Per-patient table */}
        <div style={card}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Per-Patient Upload Summary</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>Uploads</th>
                  <th style={{ padding: '6px 10px' }}>Unique Files</th>
                  <th style={{ padding: '6px 10px' }}>First Upload</th>
                  <th style={{ padding: '6px 10px' }}>Last Upload</th>
                </tr>
              </thead>
              <tbody>
                {bd.per_patient.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fafbfc' : '#fff' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{p.uploads}</td>
                    <td style={{ padding: '6px 10px' }}>{p.unique_files}</td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{p.first_upload?.slice(0, 16)}</td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{p.last_upload?.slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Format per patient */}
        <div style={card}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Format x Patient Matrix</h4>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={bd.format_patient_matrix}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="patient_id" tick={{ fontSize: 10, angle: -35 }} height={50} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4,4,0,0]} name="Uploads" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top files */}
        <div style={card}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Top Uploaded Files</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>File Name</th>
                  <th style={{ padding: '6px 10px' }}>Uploads</th>
                  <th style={{ padding: '6px 10px' }}>Patients</th>
                </tr>
              </thead>
              <tbody>
                {bd.top_files.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fafbfc' : '#fff' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{f.file_name}</td>
                    <td style={{ padding: '6px 10px' }}>{f.uploads}</td>
                    <td style={{ padding: '6px 10px' }}>{f.patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </>
    )
  }

  function renderActivity() {
    const bd = breakdown
    if (!bd?.available) return <div>No activity data available.</div>

    return (
      <>
        {/* Daily upload trend */}
        <div style={card}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Daily Upload Trend</h4>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={bd.daily_upload_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Line type="monotone" dataKey="uploads" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Daily ingest trend */}
        <div style={card}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Daily Ingest Transactions</h4>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={bd.daily_ingest_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="transactions" fill="#10b981" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly pattern */}
        <div style={card}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Hourly Upload Pattern</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={bd.hourly_pattern}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tick={{ fontSize: 11 }} label={{ value: 'Hour of Day', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#f59e0b" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent ingest events */}
        <div style={card}>
          <h4 style={{ margin: '0 0 10px', fontSize: 14, fontWeight: 600 }}>Recent Ingest Events</h4>
          <div style={{ overflowX: 'auto', maxHeight: 350, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left', position: 'sticky', top: 0, background: '#fff' }}>
                  <th style={{ padding: '5px 8px' }}>Patient</th>
                  <th style={{ padding: '5px 8px' }}>Component</th>
                  <th style={{ padding: '5px 8px' }}>Action</th>
                  <th style={{ padding: '5px 8px' }}>Actor</th>
                  <th style={{ padding: '5px 8px' }}>Detail</th>
                  <th style={{ padding: '5px 8px' }}>Time</th>
                </tr>
              </thead>
              <tbody>
                {bd.recent_ingest_events.map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fafbfc' : '#fff' }}>
                    <td style={{ padding: '5px 8px', fontWeight: 600 }}>{e.patient_id}</td>
                    <td style={{ padding: '5px 8px' }}>{e.component}</td>
                    <td style={{ padding: '5px 8px' }}>{e.action}</td>
                    <td style={{ padding: '5px 8px' }}>{e.actor}</td>
                    <td style={{ padding: '5px 8px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                    <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b', whiteSpace: 'nowrap' }}>{e.ts_local?.slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </>
    )
  }

  function renderDefinitions() {
    if (!defs?.definitions) return <div>No definitions available.</div>
    const entries = Object.entries(defs.definitions)
    return (
      <div style={card}>
        <h4 style={{ margin: '0 0 14px', fontSize: 14, fontWeight: 600 }}>Metric Definitions</h4>
        <div style={{ display: 'grid', gap: 10 }}>
          {entries.map(([key, desc]) => (
            <div key={key} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
              <strong style={{ fontSize: 13, color: '#1e293b' }}>{key}</strong>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }
}
