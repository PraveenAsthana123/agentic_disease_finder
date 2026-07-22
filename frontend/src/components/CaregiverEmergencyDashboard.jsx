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

function Badge({ text, color }) {
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
      background: color + '20', color: color
    }}>{text}</span>
  )
}

function TrainingBadge({ trained }) {
  return trained
    ? <Badge text="Trained" color="#10b981" />
    : <Badge text="Untrained" color="#ef4444" />
}

function PrimaryBadge({ primary }) {
  return primary
    ? <Badge text="Primary" color="#3b82f6" />
    : <Badge text="Secondary" color="#94a3b8" />
}

function NotifyBadge({ notify }) {
  return notify
    ? <Badge text="Auto-notify" color="#10b981" />
    : <Badge text="Manual" color="#94a3b8" />
}

function burnoutColor(score) {
  if (score < 40) return '#10b981'
  if (score < 70) return '#f59e0b'
  return '#ef4444'
}

const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#3b82f6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'caregivers', label: 'Caregivers' },
  { id: 'emergency', label: 'Emergency' },
  { id: 'definitions', label: 'Definitions' },
]

export default function CaregiverEmergencyDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/caregiver-emergency/overview`),
      axios.get(`${API_URL}/api/caregiver-emergency/breakdown`),
      axios.get(`${API_URL}/api/caregiver-emergency/definitions`),
    ]).then(([o, b, d]) => {
      setOv(o.data); setBd(b.data); setDefs(d.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading caregiver & emergency contact data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Caregiver & Emergency Contact Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        {ov?.total_caregivers} caregivers, {ov?.total_emergency_contacts} emergency contacts, {ov?.total_patients} patients | Training rate {ov?.epilepsy_training_rate}% | Avg burnout {ov?.avg_burnout_score}
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t.id ? '#2563eb' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && ov && renderOverview(ov)}
      {tab === 'caregivers' && bd && renderCaregivers(bd)}
      {tab === 'emergency' && bd && renderEmergency(bd)}
      {tab === 'definitions' && defs && renderDefinitions(defs)}
    </div>
  )
}

function renderOverview(ov) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Caregiver & Contact KPIs" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
          <KPI label="Total Caregivers" value={ov.total_caregivers} />
          <KPI label="Emergency Contacts" value={ov.total_emergency_contacts} />
          <KPI label="Training Rate" value={`${ov.epilepsy_training_rate}%`} color={ov.epilepsy_training_rate >= 80 ? '#10b981' : '#f59e0b'} />
          <KPI label="Avg Burnout" value={ov.avg_burnout_score} color={burnoutColor(ov.avg_burnout_score)} />
          <KPI label="First Aid Cert." value={`${ov.first_aid_certified_rate}%`} color="#3b82f6" />
          <KPI label="Rescue Med" value={`${ov.rescue_med_trained_rate}%`} color="#8b5cf6" />
          <KPI label="Safety Plan" value={`${ov.safety_plan_rate}%`} color="#06b6d4" />
          <KPI label="Action Plan" value={`${ov.seizure_action_plan_rate}%`} color="#10b981" />
          <KPI label="Avg Confidence" value={`${ov.avg_seizure_first_aid_confidence}/10`} color="#1e293b" />
          <KPI label="Avg Stress" value={`${ov.avg_caregiver_stress}/10`} color={ov.avg_caregiver_stress > 5 ? '#ef4444' : '#10b981'} />
          <KPI label="Avg Sleep Quality" value={`${ov.avg_sleep_quality}/10`} color={ov.avg_sleep_quality < 5 ? '#ef4444' : '#10b981'} />
          <KPI label="Patients" value={ov.total_patients} />
        </div>
      </Card>

      <Card title="Role Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.role_distribution} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" name="Count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Availability Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.availability_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(ov.availability_distribution || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Training Topic Frequency" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={ov.training_topic_frequency} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={200} tick={{ fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="value" name="Caregivers" fill="#10b981" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Burnout Score Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.burnout_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" name="Caregivers" radius={[4, 4, 0, 0]}>
              {(ov.burnout_distribution || []).map((entry, i) => {
                const c = entry.name === '81-100' || entry.name === '61-80' ? '#ef4444'
                  : entry.name === '41-60' ? '#f59e0b' : '#10b981'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Emergency Contact Relationships">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.relationship_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(ov.relationship_distribution || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderCaregivers(bd) {
  const th = { padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }
  const td = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #e2e8f0' }
  const thRed = { ...th, background: '#fef2f2', color: '#991b1b' }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {(bd.high_burnout_caregivers || []).length > 0 && (
        <Card title="High Burnout Caregivers (Score >= 70)" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thRed}>Name</th>
                  <th style={thRed}>Patient</th>
                  <th style={thRed}>Role</th>
                  <th style={thRed}>Burnout</th>
                  <th style={thRed}>Stress</th>
                  <th style={thRed}>Sleep Quality</th>
                  <th style={thRed}>Work Impact</th>
                </tr>
              </thead>
              <tbody>
                {(bd.high_burnout_caregivers || []).map((c, i) => (
                  <tr key={i}>
                    <td style={td}><strong>{c.name}</strong></td>
                    <td style={td}>{c.patient_id}</td>
                    <td style={td}>{c.role}</td>
                    <td style={td}><Badge text={c.burnout_score} color="#ef4444" /></td>
                    <td style={td}>{c.caregiver_stress}/10</td>
                    <td style={td}>{c.caregiver_sleep_quality}/10</td>
                    <td style={td}>{c.work_impact}/10</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {(bd.untrained_caregivers || []).length > 0 && (
        <Card title="Untrained Caregivers" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={th}>Name</th>
                  <th style={th}>Patient</th>
                  <th style={th}>Role</th>
                  <th style={th}>First Aid Cert.</th>
                  <th style={th}>Rescue Med</th>
                </tr>
              </thead>
              <tbody>
                {(bd.untrained_caregivers || []).map((c, i) => (
                  <tr key={i}>
                    <td style={td}><strong>{c.name}</strong></td>
                    <td style={td}>{c.patient_id}</td>
                    <td style={td}>{c.role}</td>
                    <td style={td}>{c.first_aid_certified ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                    <td style={td}>{c.rescue_med_trained ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="All Caregivers" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Patient</th>
                <th style={th}>Name</th>
                <th style={th}>Role</th>
                <th style={th}>Availability</th>
                <th style={th}>Exp. (yrs)</th>
                <th style={th}>Training</th>
                <th style={th}>First Aid</th>
                <th style={th}>Rescue Med</th>
                <th style={th}>Confidence</th>
                <th style={th}>Burnout</th>
                <th style={th}>Safety Plan</th>
                <th style={th}>Action Plan</th>
              </tr>
            </thead>
            <tbody>
              {(bd.all_caregivers || []).map((c, i) => (
                <tr key={i}>
                  <td style={td}>{c.patient_id}</td>
                  <td style={td}><strong>{c.name}</strong></td>
                  <td style={td}>{c.role}</td>
                  <td style={td}>{c.availability}</td>
                  <td style={td}>{c.experience_years}</td>
                  <td style={td}><TrainingBadge trained={c.epilepsy_training_completed} /></td>
                  <td style={td}>{c.first_aid_certified ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>{c.rescue_med_trained ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>{c.seizure_first_aid_confidence}/10</td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3, minWidth: 40 }}>
                        <div style={{ width: `${c.burnout_score}%`, height: '100%', background: burnoutColor(c.burnout_score), borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{c.burnout_score}</span>
                    </div>
                  </td>
                  <td style={td}>{c.safety_plan_exists ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>{c.seizure_action_plan_exists ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderEmergency(bd) {
  const th = { padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }
  const td = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #e2e8f0' }
  const thRed = { ...th, background: '#fef2f2', color: '#991b1b' }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {(bd.stale_contacts || []).length > 0 && (
        <Card title="Stale Emergency Contacts (Not Verified > 1 Year)" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thRed}>Contact Name</th>
                  <th style={thRed}>Patient</th>
                  <th style={thRed}>Phone</th>
                  <th style={thRed}>Relationship</th>
                  <th style={thRed}>Last Verified</th>
                  <th style={thRed}>Primary</th>
                </tr>
              </thead>
              <tbody>
                {(bd.stale_contacts || []).map((c, i) => (
                  <tr key={i}>
                    <td style={td}><strong>{c.contact_name}</strong></td>
                    <td style={td}>{c.patient_id}</td>
                    <td style={td}>{c.phone}</td>
                    <td style={td}>{c.relationship}</td>
                    <td style={td}><Badge text={c.last_verified} color="#ef4444" /></td>
                    <td style={td}><PrimaryBadge primary={c.is_primary} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="All Emergency Contacts" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Patient</th>
                <th style={th}>Contact Name</th>
                <th style={th}>Phone</th>
                <th style={th}>Email</th>
                <th style={th}>Relationship</th>
                <th style={th}>Primary</th>
                <th style={th}>Notify</th>
                <th style={th}>Last Verified</th>
              </tr>
            </thead>
            <tbody>
              {(bd.all_emergency_contacts || []).map((c, i) => (
                <tr key={i}>
                  <td style={td}>{c.patient_id}</td>
                  <td style={td}><strong>{c.contact_name}</strong></td>
                  <td style={td}>{c.phone}</td>
                  <td style={{ ...td, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.email}</td>
                  <td style={td}>{c.relationship}</td>
                  <td style={td}><PrimaryBadge primary={c.is_primary} /></td>
                  <td style={td}><NotifyBadge notify={c.notify_on_seizure} /></td>
                  <td style={td}>{c.last_verified}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Per-Patient Contact Coverage" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Patient</th>
                <th style={th}>Caregivers</th>
                <th style={th}>Emergency Contacts</th>
                <th style={th}>Trained Caregiver</th>
                <th style={th}>Primary Contact</th>
                <th style={th}>Avg Burnout</th>
              </tr>
            </thead>
            <tbody>
              {(bd.patient_summary || []).map((p, i) => (
                <tr key={i}>
                  <td style={td}>{p.patient_id}</td>
                  <td style={td}>{p.caregiver_count}</td>
                  <td style={td}>{p.emergency_contact_count}</td>
                  <td style={td}>{p.has_trained_caregiver ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>{p.has_primary_contact ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3, minWidth: 40 }}>
                        <div style={{ width: `${p.avg_burnout}%`, height: '100%', background: burnoutColor(p.avg_burnout), borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{p.avg_burnout}</span>
                    </div>
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

function renderDefinitions(defs) {
  const th = { padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }
  const td = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #e2e8f0' }
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Caregiver Roles" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Role</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.role_descriptions || []).map((r, i) => (
              <tr key={i}><td style={td}><strong>{r.role}</strong></td><td style={td}>{r.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Availability Types">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Type</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.availability_descriptions || []).map((a, i) => (
              <tr key={i}><td style={td}><strong>{a.availability}</strong></td><td style={td}>{a.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Training Topics" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Topic</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.training_topic_descriptions || []).map((t, i) => (
              <tr key={i}><td style={td}><strong>{t.topic}</strong></td><td style={td}>{t.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Score Descriptions">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Score</th><th style={th}>Range</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.score_descriptions || []).map((s, i) => (
              <tr key={i}>
                <td style={td}><strong>{s.score}</strong></td>
                <td style={td}>{s.range}</td>
                <td style={td}>{s.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Emergency Protocol Glossary" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Term</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.emergency_protocol_glossary || []).map((e, i) => (
              <tr key={i}><td style={td}><strong>{e.term}</strong></td><td style={td}>{e.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Glossary" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Term</th><th style={th}>Definition</th></tr></thead>
          <tbody>
            {(defs.glossary || []).map((g, i) => (
              <tr key={i}><td style={td}><strong>{g.term}</strong></td><td style={td}>{g.definition}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Clinical Notes" span={2}>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
          {(defs.clinical_notes || []).map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      </Card>

      {defs.data_source && (
        <Card title="Data Source" span={3}>
          <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>{defs.data_source}</p>
        </Card>
      )}
    </div>
  )
}
