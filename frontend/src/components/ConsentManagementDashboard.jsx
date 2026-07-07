import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'

const STATUS_COLORS = {
  granted: '#22c55e',
  pending: '#eab308',
  declined: '#ef4444',
  expired: '#94a3b8',
  withdrawn: '#f97316'
}
const STATUS_LABELS = {
  granted: 'Granted',
  pending: 'Pending',
  declined: 'Declined',
  expired: 'Expired',
  withdrawn: 'Withdrawn'
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  const label = STATUS_LABELS[status] || status || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function ConsentManagementDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [filterType, setFilterType] = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/consent-dashboard/overview`),
          axios.get(`${API_URL}/consent-dashboard/breakdown`),
          axios.get(`${API_URL}/consent-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Consent Management data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'types', label: 'Consent Types' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'compliance', label: 'Compliance' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const statusDistData = overview?.by_status
    ? Object.entries(overview.by_status).map(([k, v]) => ({
        name: STATUS_LABELS[k] || k, value: v, color: STATUS_COLORS[k] || '#94a3b8'
      }))
    : []

  const typeBarData = overview?.by_type
    ? Object.entries(overview.by_type).map(([k, v]) => ({
        name: k, count: typeof v === 'object' ? (v.total || 0) : v
      }))
    : []

  // Consent types for filters
  const consentTypes = breakdown?.by_type ? Object.keys(breakdown.by_type) : []

  // Filtered patient rows
  const allPatients = breakdown?.patients || []
  const filteredPatients = allPatients.filter(p => {
    const typeOk = filterType === 'all' || (p.consents && p.consents[filterType])
    const statusOk = filterStatus === 'all' || (
      p.consents && Object.values(p.consents).some(s => s === filterStatus)
    )
    return typeOk && statusOk
  })

  // Compliance matrix: patients × types
  const compliancePatients = breakdown?.patients || []
  const complianceTypes = consentTypes

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Consent Management Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Patient consent tracking — HIPAA / IRB compliance across all consent types and statuses
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Cards */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              {(overview.kpis || []).map((kpi, i) => (
                <KPI
                  key={i}
                  label={kpi.label}
                  value={fmt(kpi.value)}
                  sub={kpi.sub}
                  color={kpi.color}
                />
              ))}
            </div>
          </Card>

          {/* Pie: Status Distribution */}
          <Card title="Consent Status Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={statusDistData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {statusDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {statusDistData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Consents per Type */}
          <Card title="Consents per Type" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={typeBarData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" name="Total Consents" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Expiring Soon Alert */}
          {overview.expiring_soon && overview.expiring_soon.length > 0 && (
            <Card span={3} title="Expiring Soon">
              <div style={{
                padding: '10px 14px', background: '#fef9c3', border: '1px solid #fde047',
                borderRadius: 8, marginBottom: 12, fontSize: 13, color: '#854d0e'
              }}>
                {overview.expiring_soon.length} consent(s) expiring within 30 days
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {overview.expiring_soon.map((e, i) => (
                  <div key={i} style={{
                    padding: '6px 12px', background: '#fff7ed', border: '1px solid #fdba74',
                    borderRadius: 8, fontSize: 12
                  }}>
                    <span style={{ fontWeight: 600 }}>{e.patient_name || e.patient_id}</span>
                    <span style={{ color: '#64748b', marginLeft: 8 }}>{e.consent_type}</span>
                    <span style={{ color: '#f97316', marginLeft: 8 }}>exp: {e.expiry_date}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Recent Activity Table */}
          <Card title="Recent Activity (last 10)" span={3}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Consent Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.recent_activity || []).slice(0, 10).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_name || row.patient_id}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.consent_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{row.date}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#94a3b8' }}>{row.note || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 2: Consent Types */}
      {tab === 'types' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Bar: Consent Rate by Type */}
          <Card title="Consent Rate by Type" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={
                Object.entries(breakdown.by_type || {}).map(([k, v]) => ({
                  name: k,
                  granted: v.granted || 0,
                  pending: v.pending || 0,
                  declined: v.declined || 0,
                  expired: v.expired || 0,
                  withdrawn: v.withdrawn || 0
                }))
              }>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="granted" fill={STATUS_COLORS.granted} name="Granted" stackId="a" />
                <Bar dataKey="pending" fill={STATUS_COLORS.pending} name="Pending" stackId="a" />
                <Bar dataKey="declined" fill={STATUS_COLORS.declined} name="Declined" stackId="a" />
                <Bar dataKey="expired" fill={STATUS_COLORS.expired} name="Expired" stackId="a" />
                <Bar dataKey="withdrawn" fill={STATUS_COLORS.withdrawn} name="Withdrawn" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Type Breakdown Cards */}
          {Object.entries(breakdown.by_type || {}).map(([type, v]) => {
            const defEntry = (defs?.consent_types || []).find(d => d.type === type)
            return (
              <Card key={type} title={type}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
                  <KPI label="Granted" value={fmt(v.granted)} color={STATUS_COLORS.granted} />
                  <KPI label="Pending" value={fmt(v.pending)} color={STATUS_COLORS.pending} />
                  <KPI label="Declined" value={fmt(v.declined)} color={STATUS_COLORS.declined} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 12 }}>
                  <KPI label="Expired" value={fmt(v.expired)} color={STATUS_COLORS.expired} />
                  <KPI label="Withdrawn" value={fmt(v.withdrawn)} color={STATUS_COLORS.withdrawn} />
                </div>
                {defEntry && (
                  <div style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#475569' }}>
                    {defEntry.description}
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      )}

      {/* Tab 3: Patient Detail */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Filters */}
          <Card>
            <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
              <div>
                <label style={{ fontSize: 12, color: '#64748b', marginRight: 8 }}>Filter by Type:</label>
                <select
                  value={filterType}
                  onChange={e => setFilterType(e.target.value)}
                  style={{ fontSize: 12, padding: '4px 8px', borderRadius: 6, border: '1px solid #e2e8f0', color: '#334155' }}
                >
                  <option value="all">All Types</option>
                  {consentTypes.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
              <div>
                <label style={{ fontSize: 12, color: '#64748b', marginRight: 8 }}>Filter by Status:</label>
                <select
                  value={filterStatus}
                  onChange={e => setFilterStatus(e.target.value)}
                  style={{ fontSize: 12, padding: '4px 8px', borderRadius: 6, border: '1px solid #e2e8f0', color: '#334155' }}
                >
                  <option value="all">All Statuses</option>
                  {Object.entries(STATUS_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                </select>
              </div>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>
                Showing {filteredPatients.length} of {allPatients.length} patients
              </span>
            </div>
          </Card>

          {/* Patient Table */}
          <Card title="Patient Consent Status">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Age</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Disease</th>
                    {consentTypes.map(ct => (
                      <th key={ct} style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>{ct}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredPatients.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_name || p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>{p.age || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{p.disease || '--'}</td>
                      {consentTypes.map(ct => (
                        <td key={ct} style={{ padding: '6px 8px', textAlign: 'center' }}>
                          {p.consents && p.consents[ct]
                            ? <StatusBadge status={p.consents[ct]} />
                            : <span style={{ color: '#e2e8f0', fontSize: 11 }}>--</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 4: Compliance */}
      {tab === 'compliance' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Compliance KPIs */}
          <Card span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              {(breakdown.compliance_kpis || []).map((kpi, i) => (
                <KPI
                  key={i}
                  label={kpi.label}
                  value={fmt(kpi.value)}
                  sub={kpi.sub}
                  color={kpi.color}
                />
              ))}
            </div>
          </Card>

          {/* Compliance Matrix */}
          <Card title="Compliance Matrix — Patients x Consent Types" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 12 }}>Patient</th>
                    {complianceTypes.map(ct => (
                      <th key={ct} style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>{ct}</th>
                    ))}
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 12 }}>Complete</th>
                  </tr>
                </thead>
                <tbody>
                  {compliancePatients.map((p, i) => {
                    const total = complianceTypes.length
                    const grantedCount = complianceTypes.filter(ct => p.consents && p.consents[ct] === 'granted').length
                    const allGranted = grantedCount === total
                    return (
                      <tr key={i} style={{
                        borderBottom: '1px solid #f1f5f9',
                        background: allGranted ? '#f0fdf4' : 'transparent'
                      }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{p.patient_name || p.patient_id}</td>
                        {complianceTypes.map(ct => {
                          const status = p.consents && p.consents[ct]
                          const color = STATUS_COLORS[status] || '#e2e8f0'
                          return (
                            <td key={ct} style={{ padding: '6px 8px', textAlign: 'center' }}>
                              <span style={{
                                display: 'inline-block', width: 14, height: 14, borderRadius: 3,
                                background: color, title: status
                              }} title={status || 'N/A'} />
                            </td>
                          )
                        })}
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 12 }}>
                          {allGranted
                            ? <span style={{ color: '#22c55e', fontWeight: 700 }}>Yes</span>
                            : <span style={{ color: '#ef4444' }}>{grantedCount}/{total}</span>}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {/* Legend */}
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
              {Object.entries(STATUS_LABELS).map(([k, v]) => (
                <span key={k} style={{ fontSize: 11, color: '#475569', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 3, background: STATUS_COLORS[k] }} />
                  {v}
                </span>
              ))}
            </div>
          </Card>

          {/* Gap Analysis */}
          <Card title="Gap Analysis — Missing Consents" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Missing / Non-Granted Consents</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Gap Count</th>
                  </tr>
                </thead>
                <tbody>
                  {compliancePatients
                    .map(p => {
                      const missing = complianceTypes.filter(ct => !p.consents || p.consents[ct] !== 'granted')
                      return { ...p, missing }
                    })
                    .filter(p => p.missing.length > 0)
                    .sort((a, b) => b.missing.length - a.missing.length)
                    .map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{p.patient_name || p.patient_id}</td>
                        <td style={{ padding: '6px 8px' }}>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {p.missing.map(ct => (
                              <span key={ct} style={{
                                display: 'inline-block', padding: '1px 6px', borderRadius: 4, fontSize: 10,
                                background: '#fee2e2', color: '#ef4444', fontWeight: 600
                              }}>{ct}: <StatusBadge status={p.consents && p.consents[ct] ? p.consents[ct] : 'missing'} /></span>
                            ))}
                          </div>
                        </td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700, color: '#ef4444' }}>
                          {p.missing.length}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Consent Type Definitions */}
          {defs.consent_types && defs.consent_types.length > 0 && (
            <Card title="Consent Type Definitions">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.consent_types.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.type}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Status Definitions */}
          {defs.status_definitions && defs.status_definitions.length > 0 && (
            <Card title="Status Definitions">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.status_definitions.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', verticalAlign: 'top', width: 160 }}>
                        <StatusBadge status={d.status} />
                      </td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Regulatory Framework */}
          {defs.regulatory_framework && (
            <Card title="Regulatory Framework — HIPAA / IRB">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                {(defs.regulatory_framework.sections || []).map((sec, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                    <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>{sec.title}</h4>
                    <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{sec.description}</p>
                    {sec.requirements && (
                      <ul style={{ margin: '8px 0 0', padding: '0 0 0 16px', fontSize: 12, color: '#64748b' }}>
                        {sec.requirements.map((r, j) => <li key={j} style={{ marginBottom: 4 }}>{r}</li>)}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Glossary */}
          {defs.glossary && defs.glossary.length > 0 && (
            <Card title="Glossary">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.glossary.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.term}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* References */}
          {defs.references && (
            <Card>
              <div style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>References</h4>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#64748b' }}>
                  {defs.references.map((ref, i) => <li key={i} style={{ marginBottom: 4 }}>{ref}</li>)}
                </ul>
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
