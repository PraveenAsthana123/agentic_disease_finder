import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = '/api'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4']

const URGENCY_COLORS = {
  urgent: '#ef4444',
  standard: '#f59e0b',
  informational: '#3b82f6'
}

const TYPE_COLORS = {
  appointment_reminder: '#3b82f6',
  medication_alert: '#ef4444',
  seizure_follow_up: '#f59e0b',
  assessment_summary: '#10b981',
  wellness_check: '#8b5cf6'
}

export default function CommunicationAIDashboard() {
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
      axios.get(`${API}/communication-ai/overview`),
      axios.get(`${API}/communication-ai/breakdown`),
      axios.get(`${API}/communication-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'messages', label: 'Messages' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'templates', label: 'Templates' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Communication AI dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No communication data available.</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Communication AI</h2>
        <Badge
          text={`${overview.total_messages_generated || 0} Messages`}
          color="#3b82f6"
        />
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'messages' && renderMessages()}
      {tab === 'patients' && renderPatients()}
      {tab === 'templates' && renderTemplates()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpis = overview.kpi_cards || []
    const urgencyDist = overview.urgency_distribution || []
    const messageTypes = overview.message_types || []
    const deliveryChannels = overview.delivery_channels || []
    const messagesByDisease = overview.messages_by_disease || []
    const timeline = overview.communication_timeline || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} sub={kp.sub} color={kp.color} /></Card>
        ))}

        <Card title="Urgency Distribution" span={2}>
          {urgencyDist.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={urgencyDist} dataKey="count" nameKey="urgency" cx="50%" cy="50%" outerRadius={100} label={({ urgency, count, pct }) => `${urgency}: ${count} (${pct != null ? pct + '%' : ''})`}>
                  {urgencyDist.map((d, i) => <Cell key={i} fill={URGENCY_COLORS[d.urgency] || COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Message Types" span={2}>
          {messageTypes.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={messageTypes} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Messages" radius={[4, 4, 0, 0]}>
                  {messageTypes.map((m, i) => <Cell key={i} fill={TYPE_COLORS[m.type] || COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Delivery Channels" span={2}>
          {deliveryChannels.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={deliveryChannels} dataKey="count" nameKey="channel" cx="50%" cy="50%" outerRadius={100} label={({ channel, count, pct }) => `${channel}: ${count} (${pct != null ? pct + '%' : ''})`}>
                  {deliveryChannels.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Messages by Disease" span={2}>
          {messagesByDisease.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={messagesByDisease} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="disease" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Messages" radius={[4, 4, 0, 0]}>
                  {messagesByDisease.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Communication Timeline" span={4}>
          {timeline.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timeline} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" name="Messages" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderMessages() {
    const messages = overview.recent_messages || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Recent Messages (${messages.length})`}>
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Timestamp</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Type</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Urgency</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Content</th>
                </tr>
              </thead>
              <tbody>
                {messages.slice(0, 10).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace', fontSize: 11, whiteSpace: 'nowrap' }}>{m.timestamp || 'N/A'}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <div style={{ fontWeight: 600 }}>{m.patient_name || m.patient_id}</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>{m.patient_id}</div>
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={m.type || 'unknown'} color={TYPE_COLORS[m.type] || '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={m.urgency || 'unknown'} color={URGENCY_COLORS[m.urgency] || '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#475569', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {m.content || 'N/A'}
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

  function renderPatients() {
    const patients = breakdown?.per_patient_profiles || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Communication Profiles (${patients.length} patients)`}>
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Name</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Messages</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Last Contact</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Types Received</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Urgency Level</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Preference</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{p.patient_id}</td>
                    <td style={{ padding: '10px 12px' }}>{p.name || 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600 }}>{p.message_count != null ? p.message_count : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{p.last_contact || 'N/A'}</td>
                    <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>
                      {Array.isArray(p.types_received) ? p.types_received.map((t, ti) => (
                        <Badge key={ti} text={t} color={TYPE_COLORS[t] || '#64748b'} />
                      )) : (p.types_received || 'N/A')}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={p.urgency_level || 'unknown'} color={URGENCY_COLORS[p.urgency_level] || '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>{p.preference || 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderTemplates() {
    const templates = breakdown?.message_templates || []
    const appointments = breakdown?.appointment_communications || []
    const medications = breakdown?.medication_communications || []
    const seizures = breakdown?.seizure_communications || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Message Templates">
          {templates.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {templates.map((t, i) => (
                <div key={i} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <Badge text={t.type || 'unknown'} color={TYPE_COLORS[t.type] || '#64748b'} />
                    {t.urgency_rule && <Badge text={t.urgency_rule} color={URGENCY_COLORS[t.urgency_rule] || '#94a3b8'} />}
                  </div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, marginBottom: 8 }}>{t.sample_content || 'No sample content'}</div>
                  {t.variables && t.variables.length > 0 && (
                    <div style={{ fontSize: 11, color: '#94a3b8' }}>
                      Variables: {t.variables.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No message templates available.</p>
          )}
        </Card>

        <Card title={`Appointment Communications (${appointments.length})`}>
          {appointments.length > 0 ? (
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead style={{ position: 'sticky', top: 0 }}>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Date</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Type</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {appointments.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ fontWeight: 600 }}>{a.patient_name || a.patient_id}</div>
                        <div style={{ fontSize: 11, color: '#94a3b8' }}>{a.patient_id}</div>
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{a.appointment_date || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={a.type || 'appointment'} color="#3b82f6" />
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>{a.message || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No appointment communications available.</p>
          )}
        </Card>

        <Card title={`Medication Communications (${medications.length})`}>
          {medications.length > 0 ? (
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead style={{ position: 'sticky', top: 0 }}>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Medication</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {medications.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ fontWeight: 600 }}>{m.patient_name || m.patient_id}</div>
                        <div style={{ fontSize: 11, color: '#94a3b8' }}>{m.patient_id}</div>
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: 600, color: '#ef4444' }}>{m.medication || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>{m.message || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No medication communications available.</p>
          )}
        </Card>

        <Card title={`Seizure Communications (${seizures.length})`}>
          {seizures.length > 0 ? (
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead style={{ position: 'sticky', top: 0 }}>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Seizure Date</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Severity</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {seizures.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ fontWeight: 600 }}>{s.patient_name || s.patient_id}</div>
                        <div style={{ fontSize: 11, color: '#94a3b8' }}>{s.patient_id}</div>
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{s.seizure_date || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={s.severity || 'unknown'} color={s.severity === 'severe' ? '#ef4444' : s.severity === 'moderate' ? '#f59e0b' : '#10b981'} />
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>{s.message || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No seizure communications available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    const sections = [
      { key: 'communication_concepts', title: 'Communication Concepts' },
      { key: 'message_categories', title: 'Message Categories' },
      { key: 'delivery_methods', title: 'Delivery Methods' },
      { key: 'clinical_relevance', title: 'Clinical Relevance' },
      { key: 'remediation_strategies', title: 'Remediation Strategies' },
    ]

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {sections.map((sec, si) => {
          const items = definitions?.[sec.key] || []
          if (items.length === 0) return null
          return (
            <Card key={si} title={sec.title}>
              {sec.key === 'communication_concepts' && items.map((item, ii) => (
                <div key={ii} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.definition}</div>
                </div>
              ))}
              {sec.key === 'message_categories' && items.map((item, ii) => (
                <div key={ii} style={{ marginBottom: 12, borderBottom: ii < items.length - 1 ? '1px solid #f1f5f9' : 'none', paddingBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{item.category}</span>
                    {item.urgency && <Badge text={item.urgency} color={URGENCY_COLORS[item.urgency] || '#64748b'} />}
                  </div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.description}</div>
                  {item.triggers && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>Triggers: {item.triggers}</div>}
                </div>
              ))}
              {sec.key === 'delivery_methods' && items.map((item, ii) => (
                <div key={ii} style={{ marginBottom: 12, borderBottom: ii < items.length - 1 ? '1px solid #f1f5f9' : 'none', paddingBottom: 12 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.method}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.description}</div>
                  <div style={{ display: 'flex', gap: 16, marginTop: 4 }}>
                    {item.latency && <span style={{ fontSize: 11, color: '#94a3b8' }}>Latency: {item.latency}</span>}
                    {item.reliability && <span style={{ fontSize: 11, color: '#94a3b8' }}>Reliability: {item.reliability}</span>}
                  </div>
                </div>
              ))}
              {sec.key === 'clinical_relevance' && items.map((item, ii) => (
                <div key={ii} style={{ marginBottom: 12, borderBottom: ii < items.length - 1 ? '1px solid #f1f5f9' : 'none', paddingBottom: 12 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.standard}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}><strong>Requirement:</strong> {item.requirement}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}><strong>How Met:</strong> {item.how_met}</div>
                </div>
              ))}
              {sec.key === 'remediation_strategies' && items.map((item, ii) => (
                <div key={ii} style={{ marginBottom: 12, borderBottom: ii < items.length - 1 ? '1px solid #f1f5f9' : 'none', paddingBottom: 12 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.scenario}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}><strong>Strategy:</strong> {item.strategy}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}><strong>Action:</strong> {item.action}</div>
                </div>
              ))}
            </Card>
          )
        })}
      </div>
    )
  }
}
