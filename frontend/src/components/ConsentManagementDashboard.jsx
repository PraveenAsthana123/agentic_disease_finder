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
const STATUS_COLORS = {
  granted: '#10b981',
  pending: '#f59e0b',
  withdrawn: '#ef4444',
  declined: '#94a3b8',
  expired: '#8b5cf6'
}
const TYPE_COLORS = {
  treatment: '#3b82f6',
  research: '#8b5cf6',
  data_sharing: '#10b981',
  genetic_testing: '#f59e0b',
  video_eeg: '#ef4444',
  imaging_sharing: '#06b6d4'
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ConsentManagementDashboard() {
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
      axios.get(`${API_URL}/api/consent-management/overview`),
      axios.get(`${API_URL}/api/consent-management/breakdown`),
      axios.get(`${API_URL}/api/consent-management/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Consent Management data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Consent Management Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Consent lifecycle analytics — type distribution, status tracking, compliance rate, expiry monitoring, patient-level detail
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
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
  const typeData = Object.entries(data.consent_type_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const statusData = Object.entries(data.status_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const witnessData = Object.entries(data.witness_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const monthlyData = data.monthly_volume || []
  const matrix = data.type_status_matrix || []
  const statuses = matrix.length > 0 ? Object.keys(matrix[0]).filter(k => k !== 'consent_type') : []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
      <Card title="Total Records">
        <KPI value={data.total_records} label="consent records" color="#3b82f6" />
      </Card>
      <Card title="Total Patients">
        <KPI value={data.total_patients} label="unique patients" color="#8b5cf6" />
      </Card>
      <Card title="Compliance Rate">
        <KPI value={`${data.compliance_rate_pct}%`} label="consents in compliance"
          color={data.compliance_rate_pct >= 80 ? '#10b981' : '#f59e0b'} />
      </Card>
      <Card title="Expiring Soon">
        <KPI value={data.expiring_soon} label="within 90 days"
          color={data.expiring_soon > 0 ? '#f59e0b' : '#10b981'} />
      </Card>
      <Card title="Expired">
        <KPI value={data.expired} label="past expiry"
          color={data.expired > 0 ? '#ef4444' : '#10b981'} />
      </Card>

      <Card title="Consent Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={typeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85} label={({ name, value }) => `${name}: ${value}`}>
              {typeData.map((entry, i) => <Cell key={i} fill={TYPE_COLORS[entry.name] || COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Status Distribution" span={3}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={statusData} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={70} />
            <Tooltip />
            <Bar dataKey="value" name="Consents">
              {statusData.map((entry, i) => (
                <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Witness Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={witnessData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#06b6d4" name="Consents" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Consent Volume" span={3}>
        {monthlyData.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={monthlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="cnt" stroke="#3b82f6" strokeWidth={2} name="Consents Granted" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>

      <Card title="Type x Status Matrix" span={5}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Consent Type</th>
                {statuses.map(s => (
                  <th key={s} style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: STATUS_COLORS[s] || '#94a3b8', marginRight: 4 }} />
                    {s}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: TYPE_COLORS[row.consent_type] || '#94a3b8', marginRight: 6 }} />
                    {row.consent_type}
                  </td>
                  {statuses.map(s => (
                    <td key={s} style={{ padding: '6px 12px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', fontWeight: 600, color: STATUS_COLORS[s] || '#1e293b' }}>
                      {row[s] || 0}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Per-Patient Consent Summary (${(data.per_patient || []).length} patients)`} span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Total</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Granted</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Pending</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Withdrawn</th>
              </tr>
            </thead>
            <tbody>
              {(data.per_patient || []).map((p, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.total}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: '#10b981' }}>{p.granted}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: '#f59e0b' }}>{p.pending}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: '#ef4444' }}>{p.withdrawn}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Recent Consents (last 20)" span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Consent Type</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Granted Date</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Expiry Date</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Witness</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_consents || []).map((c, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{c.patient_id}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: TYPE_COLORS[c.consent_type] || '#94a3b8', marginRight: 6 }} />
                    {c.consent_type}
                  </td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: STATUS_COLORS[c.status] || '#94a3b8', marginRight: 4 }} />
                    {c.status}
                  </td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(c.granted_date || '').slice(0, 10)}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(c.expiry_date || '').slice(0, 10)}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{c.witness}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {(data.expiring_soon_list || []).length > 0 && (
        <Card title={`Expiring Soon (${data.expiring_soon_list.length} consents within 90 days)`} span={1}>
          <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef3c7' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Consent Type</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Expiry Date</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Days Left</th>
                </tr>
              </thead>
              <tbody>
                {data.expiring_soon_list.map((c, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#fffbeb' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{c.patient_id}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: TYPE_COLORS[c.consent_type] || '#94a3b8', marginRight: 6 }} />
                      {c.consent_type}
                    </td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(c.expiry_date || '').slice(0, 10)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', fontWeight: 600, color: (c.days_left || 0) < 30 ? '#ef4444' : '#f59e0b' }}>{c.days_left}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {(data.withdrawn_list || []).length > 0 && (
        <Card title={`Withdrawn Consents (${data.withdrawn_list.length})`} span={1}>
          <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Consent Type</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Granted Date</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Withdrawn Date</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Witness</th>
                </tr>
              </thead>
              <tbody>
                {data.withdrawn_list.map((c, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#fef2f2' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{c.patient_id}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: TYPE_COLORS[c.consent_type] || '#94a3b8', marginRight: 6 }} />
                      {c.consent_type}
                    </td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(c.granted_date || '').slice(0, 10)}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap', color: '#ef4444' }}>{(c.withdrawn_date || '').slice(0, 10)}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{c.witness}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {(data.type_detail || []).length > 0 && (
        <Card title="Per-Type Detail" span={1}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            {data.type_detail.map((td, i) => (
              <div key={i} style={{ padding: 14, background: '#f8fafc', borderRadius: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: TYPE_COLORS[td.consent_type] || COLORS[i % COLORS.length], marginRight: 8 }} />
                  <strong style={{ fontSize: 13, color: '#1e293b' }}>{td.consent_type}</strong>
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
                  Total: {td.total} | Granted: {td.granted} | Pending: {td.pending} | Withdrawn: {td.withdrawn}
                </div>
                <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                  <div style={{
                    width: `${td.granted_pct || 0}%`, height: '100%', borderRadius: 4,
                    background: TYPE_COLORS[td.consent_type] || '#3b82f6', transition: 'width 0.3s'
                  }} />
                </div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4, textAlign: 'right' }}>
                  {td.granted_pct}% granted
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {(data.glossary || []).length > 0 && (
        <Card title={`Glossary (${data.glossary.length} terms)`} span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
            {data.glossary.map((g, i) => (
              <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
                <strong style={{ color: '#1e293b' }}>{g.term}</strong>
                <div style={{ color: '#64748b', marginTop: 2 }}>{g.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Consent Types">
        {(data.consent_types || []).map((ct, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.consent_types.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: TYPE_COLORS[ct.type] || '#94a3b8', marginRight: 6 }} />
            <strong>{ct.type}</strong>
            <div style={{ color: '#64748b', marginTop: 2, marginLeft: 16 }}>{ct.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Statuses">
        {(data.statuses || []).map((s, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.statuses.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: STATUS_COLORS[s.status] || '#94a3b8', marginRight: 6 }} />
            <strong>{s.status}</strong>
            <div style={{ color: '#64748b', marginTop: 2, marginLeft: 16 }}>{s.description}</div>
          </div>
        ))}
      </Card>

      {(data.compliance_notes || []).length > 0 && (
        <Card title="Compliance Notes" span={2}>
          {data.compliance_notes.map((n, i) => (
            <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.compliance_notes.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
              <strong style={{ color: '#1e293b' }}>{n.title || n.note_title}</strong>
              <div style={{ color: '#64748b', marginTop: 2 }}>{n.detail || n.note_detail}</div>
            </div>
          ))}
        </Card>
      )}
    </div>
  )
}
