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

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const OUTCOME_COLORS = {
  'caregiver-responded': '#10b981',
  'resolved-home': '#3b82f6',
  'er-visit': '#f59e0b',
  'ems-dispatched': '#ef4444',
  'false-alarm': '#94a3b8'
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Events & Contacts' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EmergencySOSDashboard() {
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
      axios.get(`${API_URL}/api/emergency-sos/overview`),
      axios.get(`${API_URL}/api/emergency-sos/breakdown`),
      axios.get(`${API_URL}/api/emergency-sos/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Emergency SOS data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Emergency SOS Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Emergency alert analytics — SOS events, response times, outcomes, contact coverage, preparedness tracking
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
  const eventTypeData = Object.entries(data.event_type_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const triggerData = Object.entries(data.trigger_method_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const outcomeData = Object.entries(data.outcome_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const relData = Object.entries(data.contacts?.relationship_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const rs = data.response_time_stats || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total SOS Events">
        <KPI value={data.total_events} label="emergency alerts" sub={`${data.patients_with_events} patients`} color="#ef4444" />
      </Card>
      <Card title="Avg Response Time">
        <KPI value={`${rs.avg_seconds}s`} label="alert to acknowledgement" sub={`${rs.pct_under_2min}% under 2 min`} color="#f59e0b" />
      </Card>
      <Card title="Responder Notified">
        <KPI value={`${data.responder_notified_pct}%`} label="notification rate" color="#10b981" />
      </Card>
      <Card title="False Alarm Rate">
        <KPI value={`${data.false_alarm_rate_pct}%`} label="of all events"
          sub={data.false_alarm_rate_pct > 15 ? 'Above 15% target' : 'Within target'}
          color={data.false_alarm_rate_pct > 15 ? '#ef4444' : '#10b981'} />
      </Card>

      <Card title="Event Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={eventTypeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85} label={({ name, value }) => `${name}: ${value}`}>
              {eventTypeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Outcome Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={outcomeData} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={110} />
            <Tooltip />
            <Bar dataKey="value" name="Events">
              {outcomeData.map((entry, i) => (
                <Cell key={i} fill={OUTCOME_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Trigger Method Breakdown" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={triggerData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Events" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Response Time Range" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, padding: '20px 0' }}>
          <KPI value={`${rs.min_seconds}s`} label="Fastest" color="#10b981" />
          <KPI value={`${rs.avg_seconds}s`} label="Average" color="#f59e0b" />
          <KPI value={`${rs.max_seconds}s`} label="Slowest" color="#ef4444" />
        </div>
        <div style={{ fontSize: 12, color: '#64748b', textAlign: 'center', marginTop: 4 }}>
          Location shared: {data.location_sharing_pct}% of events
        </div>
      </Card>

      <Card title="Monthly Trend" span={2}>
        {data.monthly_trend?.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data.monthly_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="cnt" stroke="#ef4444" strokeWidth={2} name="SOS Events" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>

      <Card title="Emergency Contact Coverage" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
          <KPI value={data.contacts?.total} label="Total Contacts" color="#3b82f6" />
          <KPI value={data.contacts?.patients_covered} label="Patients Covered" color="#10b981" />
          <KPI value={`${data.contacts?.seizure_notify_pct}%`} label="Seizure Notify" color="#8b5cf6" />
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={relData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Contacts" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Response Time by Event Type" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.response_by_event_type || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="event_type" tick={{ fontSize: 11 }} />
            <YAxis label={{ value: 'seconds', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="avg_rt" fill="#f59e0b" name="Avg Response (s)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="min_rt" fill="#10b981" name="Min (s)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="max_rt" fill="#ef4444" name="Max (s)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Trigger Method vs Outcome Cross-Tab" span={1}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Trigger</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Outcome</th>
                <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {(data.trigger_outcome_crosstab || []).map((r, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{r.trigger_method}</td>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: OUTCOME_COLORS[r.outcome] || '#94a3b8', marginRight: 6 }} />
                    {r.outcome}
                  </td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{r.cnt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title={`Per-Patient Event Summary (${(data.patient_events || []).length} patients)`} span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Events</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Avg RT (s)</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>False Alarms</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>ER Visits</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>EMS</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Loc Shared</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Contacts</th>
              </tr>
            </thead>
            <tbody>
              {(data.patient_events || []).map((p, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.total_events}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.avg_response}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: p.false_alarms > 0 ? '#94a3b8' : undefined }}>{p.false_alarms}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: p.er_visits > 0 ? '#f59e0b' : undefined }}>{p.er_visits}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: p.ems_dispatched > 0 ? '#ef4444' : undefined }}>{p.ems_dispatched}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.location_shared_count}/{p.total_events}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: p.contact_count === 0 ? '#ef4444' : '#10b981' }}>{p.contact_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Recent SOS Events (last 30)" span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Type</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Trigger</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>RT (s)</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Outcome</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_events || []).map((e, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(e.event_date || '').slice(0, 10)}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{e.patient_id}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{e.event_type}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{e.trigger_method}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{e.response_time_seconds}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: OUTCOME_COLORS[e.outcome] || '#94a3b8', marginRight: 4 }} />
                    {e.outcome}
                  </td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {(data.stale_contacts || []).length > 0 && (
        <Card title={`Contacts Needing Re-Verification (${data.stale_contacts.length})`} span={1}>
          <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef3c7' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Contact</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Relationship</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Last Verified</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Days Ago</th>
                </tr>
              </thead>
              <tbody>
                {data.stale_contacts.map((c, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#fffbeb' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{c.patient_id}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{c.contact_name}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{c.relationship}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{c.last_verified}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: c.days_since_verified > 365 ? '#ef4444' : '#f59e0b', fontWeight: 600 }}>{c.days_since_verified}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title={`Clinical Glossary (${(data.glossary || []).length} terms)`} span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {(data.glossary || []).map((g, i) => (
            <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
              <strong style={{ color: '#1e293b' }}>{g.term}</strong>
              <div style={{ color: '#64748b', marginTop: 2 }}>{g.definition}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Event Types">
        {(data.event_types || []).map((e, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.event_types.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#ef4444' }}>{e.type}</strong>
            <div style={{ color: '#64748b', marginTop: 2 }}>{e.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Trigger Methods">
        {(data.trigger_methods || []).map((t, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.trigger_methods.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#3b82f6' }}>{t.method}</strong>
            <div style={{ color: '#64748b', marginTop: 2 }}>{t.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Outcome Categories">
        {(data.outcomes || []).map((o, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.outcomes.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: OUTCOME_COLORS[o.outcome] || '#94a3b8', marginRight: 6 }} />
            <strong>{o.outcome}</strong>
            <div style={{ color: '#64748b', marginTop: 2, marginLeft: 16 }}>{o.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Preparedness Targets">
        {(data.preparedness_metrics || []).map((m, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.preparedness_metrics.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#1e293b' }}>{m.metric}</strong>
            <div style={{ color: '#10b981', marginTop: 2 }}>Target: {m.target}</div>
          </div>
        ))}
      </Card>
    </div>
  )
}
