import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316', '#14b8a6']

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

function SensitivityBadge({ level }) {
  const colors = {
    'HIGH': { bg: '#fef2f2', fg: '#991b1b' },
    'MEDIUM': { bg: '#fff7ed', fg: '#9a3412' },
    'LOW': { bg: '#ecfdf5', fg: '#065f46' },
  }
  const c = colors[level] || { bg: '#f1f5f9', fg: '#475569' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: c.bg, color: c.fg
    }}>{level}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'actors', label: 'Actor Access' },
  { id: 'scan', label: 'PHI Scan' },
  { id: 'definitions', label: 'Definitions' },
]

export default function DataPrivacyDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/data-privacy/overview`),
      axios.get(`${API_URL}/data-privacy/breakdown`),
      axios.get(`${API_URL}/data-privacy/definitions`),
    ])
      .then(([oRes, bRes, dRes]) => {
        setOverview(oRes.data)
        setBreakdown(bRes.data)
        setDefs(dRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Data Privacy...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40 }}>Privacy data not available</div>

  const k = overview.kpis || {}

  const piiFieldData = (overview.pii_field_distribution || []).map((f, i) => ({
    ...f, name: f.field, value: f.non_null_count, fill: COLORS[i % COLORS.length]
  }))

  const componentData = (overview.phi_access_by_component || []).slice(0, 12).map((c, i) => ({
    ...c, fill: COLORS[i % COLORS.length]
  }))

  const sensitivityData = (() => {
    const sens = (breakdown?.component_sensitivity || [])
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0 }
    sens.forEach(s => { counts[s.sensitivity] = (counts[s.sensitivity] || 0) + s.events })
    return Object.entries(counts).map(([level, count]) => ({
      name: level, value: count,
      fill: level === 'HIGH' ? '#ef4444' : level === 'MEDIUM' ? '#f59e0b' : '#10b981'
    }))
  })()

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Data Privacy Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        PII/PHI exposure tracking, de-identification coverage, actor access audit, and conversation PHI scanning ({k.phi_access_events} PHI access events)
      </p>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
              background: tab === t.id ? '#1e293b' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#475569',
            }}
          >{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="PII Fields Exposed" value={k.total_pii_fields_exposed} color="#ef4444" /></Card>
            <Card><KPI label="Patients w/ PII" value={k.patients_with_pii} sub={`of ${k.total_patients} total`} color="#f59e0b" /></Card>
            <Card><KPI label="PHI Access Events" value={k.phi_access_events} color="#3b82f6" /></Card>
            <Card><KPI label="Unique PHI Actors" value={k.unique_phi_actors} color="#8b5cf6" /></Card>
            <Card><KPI label="De-ID Coverage" value={`${k.deidentification_coverage_pct}%`} sub="master pipeline" color="#10b981" /></Card>
            <Card><KPI label="Conv. PHI Leaks" value={k.conversation_phi_messages} sub={`${k.conversation_phi_rate_pct}% of messages`} color="#ec4899" /></Card>
            <Card><KPI label="PII Exposure" value={k.patients_with_pii > 0 ? 'Active' : 'None'} color={k.patients_with_pii > 0 ? '#ef4444' : '#10b981'} /></Card>
            <Card><KPI label="Privacy Risk" value={k.deidentification_coverage_pct < 50 ? 'Elevated' : 'Normal'} color={k.deidentification_coverage_pct < 50 ? '#ef4444' : '#10b981'} /></Card>
          </div>

          {/* PII field distribution + Sensitivity distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="PII Field Distribution">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={piiFieldData} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="field" fontSize={12} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="non_null_count" name="Non-null Records" radius={[4, 4, 0, 0]}>
                    {piiFieldData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Component Sensitivity Distribution">
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={sensitivityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
                    label={({ name, value }) => `${name}: ${value}`}>
                    {sensitivityData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* PHI access by component */}
          <Card title="PHI Access by Component (top 12)">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={componentData} layout="vertical" margin={{ left: 120, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="component" fontSize={11} width={120} />
                <Tooltip />
                <Bar dataKey="events" fill="#3b82f6" radius={[0, 4, 4, 0]} name="PHI Events" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Daily PHI access trend */}
          <div style={{ marginTop: 16 }}>
            <Card title="Daily PHI Access Trend">
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={overview.daily_phi_trend || []} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={11} />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="events" stroke="#ef4444" fill="#fef2f2" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* PHI access by action */}
          <div style={{ marginTop: 16 }}>
            <Card title="PHI Access by Action Type">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={(overview.phi_access_by_action || []).slice(0, 10)} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="action" fontSize={11} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="events" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Events" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── Patient Profiles Tab ── */}
      {tab === 'patients' && breakdown && (
        <>
          <Card title={`Patient Privacy Profiles (${(breakdown.patient_profiles || []).length} patients with data access)`}>
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Patient ID', 'Access Count', 'Components', 'Actors'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_profiles || []).slice(0, 40).map(p => (
                    <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ color: p.access_count > 10 ? '#ef4444' : '#1e293b', fontWeight: p.access_count > 10 ? 700 : 400 }}>
                        {p.access_count}
                      </td>
                      <td>
                        {(p.components || []).slice(0, 5).map(c => (
                          <span key={c} style={{
                            display: 'inline-block', padding: '1px 8px', borderRadius: 8,
                            background: '#f1f5f9', margin: '0 4px 4px 0', fontSize: 11
                          }}>{c}</span>
                        ))}
                        {(p.components || []).length > 5 && <span style={{ fontSize: 11, color: '#94a3b8' }}>+{p.components.length - 5} more</span>}
                      </td>
                      <td>
                        {(p.actors || []).map(a => (
                          <span key={a} style={{
                            display: 'inline-block', padding: '1px 8px', borderRadius: 8,
                            background: '#ecfdf5', margin: '0 4px 4px 0', fontSize: 11
                          }}>{a}</span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Upload privacy */}
          <div style={{ marginTop: 16 }}>
            <Card title={`File Upload Privacy (${(breakdown.upload_privacy || []).length} patients with uploaded files)`}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginTop: 8 }}>
                {(breakdown.upload_privacy || []).map(u => (
                  <div key={u.patient_id} style={{
                    padding: 12, background: '#fef2f2', borderRadius: 8, fontSize: 12
                  }}>
                    <div style={{ fontWeight: 700, color: '#1e293b' }}>{u.patient_id}</div>
                    <div style={{ color: '#64748b', marginTop: 4 }}>{u.file_count} file{u.file_count > 1 ? 's' : ''} uploaded</div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── Actor Access Tab ── */}
      {tab === 'actors' && breakdown && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {(breakdown.actor_phi_matrix || []).map(a => (
              <Card key={a.actor} title={`${a.actor} (${a.total_patients} patient${a.total_patients > 1 ? 's' : ''})`}>
                <div style={{ fontSize: 12, lineHeight: 2 }}>
                  {(a.patients || []).slice(0, 10).map(p => (
                    <span key={p.patient_id} style={{
                      display: 'inline-block', padding: '2px 10px', borderRadius: 8,
                      background: p.access_count > 5 ? '#fef2f2' : '#f1f5f9',
                      margin: '0 4px 4px 0', fontSize: 11
                    }}>
                      {p.patient_id} ({p.access_count})
                    </span>
                  ))}
                  {(a.patients || []).length > 10 && (
                    <span style={{ fontSize: 11, color: '#94a3b8' }}>+{a.patients.length - 10} more</span>
                  )}
                </div>
              </Card>
            ))}
          </div>

          {/* Component sensitivity */}
          <div style={{ marginTop: 16 }}>
            <Card title={`Component Data Sensitivity (${(breakdown.component_sensitivity || []).length} components)`}>
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['Component', 'Events', 'Sensitivity'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.component_sensitivity || []).map(c => (
                      <tr key={c.component} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{c.component}</td>
                        <td>{c.events}</td>
                        <td><SensitivityBadge level={c.sensitivity} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── PHI Scan Tab ── */}
      {tab === 'scan' && breakdown && (
        <>
          {/* Conversation PHI scan */}
          <Card title={`Conversation PHI Scan (${(breakdown.conversation_phi_scan || []).length} message${(breakdown.conversation_phi_scan || []).length !== 1 ? 's' : ''} with patient ID references)`}>
            {(breakdown.conversation_phi_scan || []).length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#10b981', fontSize: 14, fontWeight: 600 }}>
                No PHI leakage detected in conversation logs
              </div>
            ) : (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['#', 'Role', 'Patient IDs Found', 'Text Preview', 'Timestamp'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.conversation_phi_scan || []).map(m => (
                      <tr key={m.id} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#94a3b8' }}>{m.id}</td>
                        <td style={{ fontWeight: 600 }}>{m.role}</td>
                        <td>
                          {(m.patient_ids || []).map(pid => (
                            <span key={pid} style={{
                              display: 'inline-block', padding: '1px 8px', borderRadius: 8,
                              background: '#fef2f2', color: '#991b1b', margin: '0 4px 4px 0', fontSize: 11, fontWeight: 600
                            }}>{pid}</span>
                          ))}
                        </td>
                        <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#64748b' }}>
                          {m.text_preview}
                        </td>
                        <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(m.timestamp || '').slice(0, 19)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Recent PHI access log */}
          <div style={{ marginTop: 16 }}>
            <Card title={`Recent PHI Access Log (last ${(breakdown.recent_phi_log || []).length} events)`}>
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['#', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.recent_phi_log || []).map(r => (
                      <tr key={r.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#94a3b8' }}>{r.id}</td>
                        <td>{r.patient_id}</td>
                        <td style={{ fontWeight: 600, color: '#3b82f6' }}>{r.component}</td>
                        <td>{r.action}</td>
                        <td>{r.actor}</td>
                        <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#64748b' }}>{r.detail}</td>
                        <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(r.timestamp || '').slice(0, 19)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs?.definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          {Object.entries(defs.definitions).map(([section, items]) => (
            <Card key={section} title={section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                {Array.isArray(items) ? items.map((item, i) => {
                  const term = item.term || item.metric || item.standard || item.action
                  const desc = item.definition || item.description || item.requirement
                  return (
                    <div key={i} style={{ marginBottom: 8 }}>
                      <strong style={{ color: '#1e293b' }}>{term}:</strong>{' '}
                      <span style={{ color: '#475569' }}>{desc}</span>
                    </div>
                  )
                }) : Object.entries(items).map(([term, desc]) => (
                  <div key={term} style={{ marginBottom: 8 }}>
                    <strong style={{ color: '#1e293b' }}>{term}:</strong>{' '}
                    <span style={{ color: '#475569' }}>{desc}</span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
