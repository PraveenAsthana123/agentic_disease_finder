import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'caregiver_training', label: 'Caregiver Training' },
  { id: 'caregiver_burden', label: 'Caregiver Burden' },
  { id: 'patient_detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EmergencyCaregiverDashboard() {
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
      axios.get(`${API_URL}/api/emergency-caregiver/overview`),
      axios.get(`${API_URL}/api/emergency-caregiver/breakdown`),
      axios.get(`${API_URL}/api/emergency-caregiver/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Emergency Contact & Caregiver Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Emergency Contact & Caregiver Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Emergency contacts, caregiver training status, burden monitoring, safety plans, and rescue medication readiness
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI Row 1 */}
          <Card title="Total Patients"><KPI value={ov.total_patients} label="Patients" color="#3b82f6" /></Card>
          <Card title="Emergency Contacts"><KPI value={ov.total_emergency_contacts} label="Total contacts" color="#8b5cf6" /></Card>
          <Card title="Total Caregivers"><KPI value={ov.total_caregivers} label="Active caregivers" color="#10b981" /></Card>
          <Card title="Safety Plan Coverage"><KPI value={ov.pct_with_safety_plan != null ? `${ov.pct_with_safety_plan.toFixed(1)}%` : '--'} label="With safety plan" color="#f59e0b" /></Card>

          {/* KPI Row 2 */}
          <Card title="First Aid Certified"><KPI value={ov.pct_first_aid_certified != null ? `${ov.pct_first_aid_certified.toFixed(1)}%` : '--'} label="Certified caregivers" color="#2ecc71" /></Card>
          <Card title="Rescue Med Trained"><KPI value={ov.pct_rescue_med_trained != null ? `${ov.pct_rescue_med_trained.toFixed(1)}%` : '--'} label="Trained caregivers" color="#06b6d4" /></Card>
          <Card title="Avg Caregiver Stress"><KPI value={ov.avg_caregiver_stress != null ? ov.avg_caregiver_stress.toFixed(1) : '--'} label="Stress (1-10)" color="#e74c3c" /></Card>
          <Card title="Avg Burnout Score"><KPI value={ov.avg_burnout_score != null ? ov.avg_burnout_score.toFixed(1) : '--'} label="Burnout (1-10)" color="#ec4899" /></Card>

          {/* Relationship Distribution Pie Chart */}
          <Card title="Relationship Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.relationship_distribution || []} dataKey="count" nameKey="relationship" cx="50%" cy="50%" outerRadius={80} label={e => `${e.relationship}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.relationship_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Role Distribution Pie Chart */}
          <Card title="Role Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.role_distribution || []} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={80} label={e => `${e.role}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.role_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Availability Breakdown Bar Chart */}
          <Card title="Availability Breakdown" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.availability_breakdown || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="availability" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
                  {(ov.availability_breakdown || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Burden Distribution Bar Chart */}
          <Card title="Burden Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.burden_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
                  {(ov.burden_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* CAREGIVER TRAINING TAB */}
      {tab === 'caregiver_training' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Training Completion Rate">
            <div style={{ display: 'flex', gap: 24, marginBottom: 16 }}>
              <KPI value={bd.training_completion_rate != null ? `${bd.training_completion_rate.toFixed(1)}%` : '--'} label="Overall completion" color="#2ecc71" />
            </div>
          </Card>

          <Card title="Caregiver Training Status">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Caregiver ID', 'Patient ID', 'Epilepsy Training', 'First Aid Certified', 'Rescue Med Trained', 'Seizure First Aid Confidence'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.caregivers || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(c.caregiver_id)}</td>
                      <td style={{ padding: '6px' }}>{fmt(c.patient_id)}</td>
                      <td style={{ padding: '6px' }}>
                        <Badge
                          text={c.epilepsy_training ? 'Yes' : 'No'}
                          color={c.epilepsy_training ? '#2ecc71' : '#e74c3c'}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <Badge
                          text={c.first_aid_certified ? 'Yes' : 'No'}
                          color={c.first_aid_certified ? '#2ecc71' : '#e74c3c'}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <Badge
                          text={c.rescue_med_trained ? 'Yes' : 'No'}
                          color={c.rescue_med_trained ? '#2ecc71' : '#e74c3c'}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        {c.seizure_first_aid_confidence != null ? (
                          <Badge
                            text={`${c.seizure_first_aid_confidence}/10`}
                            color={c.seizure_first_aid_confidence >= 7 ? '#2ecc71' : c.seizure_first_aid_confidence >= 4 ? '#f59e0b' : '#e74c3c'}
                          />
                        ) : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* CAREGIVER BURDEN TAB */}
      {tab === 'caregiver_burden' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Caregiver Burden Metrics">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Caregiver ID', 'Patient ID', 'Stress', 'Sleep Quality', 'Work Impact', 'Burnout Score'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.caregivers || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(c.caregiver_id)}</td>
                      <td style={{ padding: '6px' }}>{fmt(c.patient_id)}</td>
                      <td style={{ padding: '6px' }}>
                        <Badge
                          text={c.stress != null ? c.stress.toFixed(1) : '--'}
                          color={c.stress >= 7 ? '#e74c3c' : c.stress >= 4 ? '#f59e0b' : '#2ecc71'}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <Badge
                          text={c.sleep_quality != null ? c.sleep_quality.toFixed(1) : '--'}
                          color={c.sleep_quality >= 7 ? '#2ecc71' : c.sleep_quality >= 4 ? '#f59e0b' : '#e74c3c'}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <Badge
                          text={c.work_impact != null ? c.work_impact.toFixed(1) : '--'}
                          color={c.work_impact >= 7 ? '#e74c3c' : c.work_impact >= 4 ? '#f59e0b' : '#2ecc71'}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <Badge
                          text={c.burnout_score != null ? c.burnout_score.toFixed(1) : '--'}
                          color={c.burnout_score >= 7 ? '#e74c3c' : c.burnout_score >= 4 ? '#f59e0b' : '#2ecc71'}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Burden Distribution Chart */}
          <Card title="Burden Level Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov.burden_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
                  {(ov.burden_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patient_detail' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {(bd.patients || []).slice(0, 5).map((p, pi) => (
            <Card key={pi} title={`Patient ${p.patient_id}`}>
              {/* Stats Summary */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#3b82f6' }}>{fmt(p.emergency_contact_count)}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Emergency Contacts</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#10b981' }}>{fmt(p.caregiver_count)}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Caregivers</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#f59e0b' }}>
                    <Badge
                      text={p.has_safety_plan ? 'Yes' : 'No'}
                      color={p.has_safety_plan ? '#2ecc71' : '#e74c3c'}
                    />
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Safety Plan</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#8b5cf6' }}>
                    <Badge
                      text={p.rescue_med_available ? 'Yes' : 'No'}
                      color={p.rescue_med_available ? '#2ecc71' : '#e74c3c'}
                    />
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Rescue Med Available</div>
                </div>
              </div>

              {/* Emergency Contacts Table */}
              <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Emergency Contacts</h4>
              {(p.emergency_contacts || []).length > 0 ? (
                <div style={{ overflowX: 'auto', marginBottom: 16 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: '#f1f5f9' }}>
                        {['Name', 'Relationship', 'Phone', 'Availability', 'Priority'].map(h => (
                          <th key={h} style={{ padding: '6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(p.emergency_contacts || []).map((ec, ei) => (
                        <tr key={ei} style={{ borderBottom: '1px solid #e2e8f0' }}>
                          <td style={{ padding: '4px 6px', fontWeight: 600 }}>{fmt(ec.name)}</td>
                          <td style={{ padding: '4px 6px' }}>{fmt(ec.relationship)}</td>
                          <td style={{ padding: '4px 6px' }}>{fmt(ec.phone)}</td>
                          <td style={{ padding: '4px 6px' }}>{fmt(ec.availability)}</td>
                          <td style={{ padding: '4px 6px' }}>
                            <Badge
                              text={fmt(ec.priority)}
                              color={ec.priority === 'Primary' ? '#3b82f6' : '#64748b'}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: '#94a3b8', fontSize: 12, marginBottom: 16 }}>No emergency contacts recorded.</p>
              )}

              {/* Caregivers Table */}
              <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Caregivers</h4>
              {(p.caregivers || []).length > 0 ? (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: '#f1f5f9' }}>
                        {['Name', 'Role', 'First Aid', 'Rescue Med Trained', 'Stress', 'Burnout'].map(h => (
                          <th key={h} style={{ padding: '6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(p.caregivers || []).map((cg, ci) => (
                        <tr key={ci} style={{ borderBottom: '1px solid #e2e8f0' }}>
                          <td style={{ padding: '4px 6px', fontWeight: 600 }}>{fmt(cg.name)}</td>
                          <td style={{ padding: '4px 6px' }}>{fmt(cg.role)}</td>
                          <td style={{ padding: '4px 6px' }}>
                            <Badge
                              text={cg.first_aid_certified ? 'Yes' : 'No'}
                              color={cg.first_aid_certified ? '#2ecc71' : '#e74c3c'}
                            />
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            <Badge
                              text={cg.rescue_med_trained ? 'Yes' : 'No'}
                              color={cg.rescue_med_trained ? '#2ecc71' : '#e74c3c'}
                            />
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {cg.stress != null ? (
                              <Badge
                                text={cg.stress.toFixed(1)}
                                color={cg.stress >= 7 ? '#e74c3c' : cg.stress >= 4 ? '#f59e0b' : '#2ecc71'}
                              />
                            ) : '--'}
                          </td>
                          <td style={{ padding: '4px 6px' }}>
                            {cg.burnout_score != null ? (
                              <Badge
                                text={cg.burnout_score.toFixed(1)}
                                color={cg.burnout_score >= 7 ? '#e74c3c' : cg.burnout_score >= 4 ? '#f59e0b' : '#2ecc71'}
                              />
                            ) : '--'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: '#94a3b8', fontSize: 12 }}>No caregivers recorded for this patient.</p>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Emergency Contact & Caregiver Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
