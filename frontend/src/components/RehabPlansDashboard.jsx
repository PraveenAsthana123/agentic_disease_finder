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

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const STATUS_COLORS = {
  active: '#3b82f6',
  completed: '#10b981',
  on_hold: '#f59e0b',
  discontinued: '#ef4444'
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Patients & Categories' },
  { id: 'definitions', label: 'Definitions' },
]

export default function RehabPlansDashboard() {
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
      axios.get(`${API_URL}/api/rehab-plans/overview`),
      axios.get(`${API_URL}/api/rehab-plans/breakdown`),
      axios.get(`${API_URL}/api/rehab-plans/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Rehabilitation Plans data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Rehabilitation Plans Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Rehab goal-tracking analytics — progress monitoring, session completion, goal categories, therapist notes, patient outcomes
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
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total Plans">
        <KPI value={data.total_plans} label="rehabilitation plans" sub={`${data.total_patients} patients`} color="#3b82f6" />
      </Card>
      <Card title="Avg Progress">
        <KPI value={`${data.avg_progress}%`} label="mean goal progress" color="#10b981" />
      </Card>
      <Card title="Completion Rate">
        <KPI value={`${data.completion_rate}%`} label="plans completed" color="#8b5cf6" />
      </Card>
      <Card title="Session Rate">
        <KPI value={`${data.avg_session_rate}%`} label="sessions completed"
          sub={`${data.total_sessions_completed} of ${data.total_sessions_planned}`} color="#f59e0b" />
      </Card>

      <Card title="Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={data.status_dist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85}
              label={({ name, value }) => `${name}: ${value}`}>
              {(data.status_dist || []).map((entry, i) => (
                <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Goal Category Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.category_dist} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={110} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Plans" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Progress Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.progress_dist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Plans" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Avg Progress by Category" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.category_progress} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={110} />
            <Tooltip />
            <Bar dataKey="avg_progress" fill="#10b981" name="Avg Progress %" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Category × Status" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.category_status}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="category" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="active" stackId="a" fill={STATUS_COLORS.active} name="Active" />
            <Bar dataKey="completed" stackId="a" fill={STATUS_COLORS.completed} name="Completed" />
            <Bar dataKey="on_hold" stackId="a" fill={STATUS_COLORS.on_hold} name="On Hold" />
            <Bar dataKey="discontinued" stackId="a" fill={STATUS_COLORS.discontinued} name="Discontinued" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Trend" span={2}>
        {data.monthly_trend?.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.monthly_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="new_plans" stroke="#3b82f6" strokeWidth={2} name="New Plans" dot={{ r: 3 }} />
              <Line yAxisId="left" type="monotone" dataKey="completed" stroke="#10b981" strokeWidth={2} name="Completed" dot={{ r: 3 }} />
              <Line yAxisId="right" type="monotone" dataKey="avg_progress" stroke="#f59e0b" strokeWidth={2} name="Avg Progress %" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Per-Patient Summary">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Total</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Active</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Completed</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>On Hold</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Disc.</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Avg Progress</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Session Rate</th>
              </tr>
            </thead>
            <tbody>
              {(data.per_patient || []).map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 12px', fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{p.total}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', color: '#3b82f6' }}>{p.active}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', color: '#10b981' }}>{p.completed}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', color: '#f59e0b' }}>{p.on_hold}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', color: '#ef4444' }}>{p.discontinued}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
                      <div style={{ width: 60, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${p.avg_progress}%`, height: '100%', background: p.avg_progress >= 75 ? '#10b981' : p.avg_progress >= 50 ? '#f59e0b' : '#ef4444', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11 }}>{p.avg_progress}%</span>
                    </div>
                  </td>
                  <td style={{ padding: '6px 12px', fontSize: 11, color: '#64748b' }}>{p.session_rate}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Per-Category Detail">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Category</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Total</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Active</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Completed</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Avg Progress</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Avg Sessions Done</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Avg Sessions Planned</th>
              </tr>
            </thead>
            <tbody>
              {(data.per_category || []).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 12px', fontWeight: 600, color: '#1e293b' }}>{c.goal_category}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{c.total}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', color: '#3b82f6' }}>{c.active}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', color: '#10b981' }}>{c.completed}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{c.avg_progress}%</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{c.avg_sessions_done}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{c.avg_sessions_planned}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {data.high_performers?.length > 0 && (
        <Card title="High Performers (Active, Progress >= 80%)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f0fdf4' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Category</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Goal</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Progress</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Sessions</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Notes</th>
                </tr>
              </thead>
              <tbody>
                {data.high_performers.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 12px' }}>{p.goal_category}</td>
                    <td style={{ padding: '6px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.goal_description}</td>
                    <td style={{ padding: '6px 12px', textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{p.progress_pct}%</td>
                    <td style={{ padding: '6px 12px', textAlign: 'center' }}>{p.sessions_completed}/{p.sessions_planned}</td>
                    <td style={{ padding: '6px 12px', fontSize: 11, color: '#64748b', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.therapist_notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {data.low_progress?.length > 0 && (
        <Card title="Low Progress (Active, Progress < 25%)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Category</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Goal</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Progress</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Sessions</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Notes</th>
                </tr>
              </thead>
              <tbody>
                {data.low_progress.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 12px' }}>{p.goal_category}</td>
                    <td style={{ padding: '6px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.goal_description}</td>
                    <td style={{ padding: '6px 12px', textAlign: 'center', color: '#ef4444', fontWeight: 600 }}>{p.progress_pct}%</td>
                    <td style={{ padding: '6px 12px', textAlign: 'center' }}>{p.sessions_completed}/{p.sessions_planned}</td>
                    <td style={{ padding: '6px 12px', fontSize: 11, color: '#64748b', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.therapist_notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {data.attention_plans?.length > 0 && (
        <Card title="Attention Needed (On Hold / Discontinued)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fffbeb' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Category</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Goal</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Progress</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Notes</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Last Updated</th>
                </tr>
              </thead>
              <tbody>
                {data.attention_plans.slice(0, 30).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 12px' }}>{p.goal_category}</td>
                    <td style={{ padding: '6px 12px', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.goal_description}</td>
                    <td style={{ padding: '6px 12px', textAlign: 'center' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
                        background: p.status === 'on_hold' ? '#fef3c7' : '#fee2e2',
                        color: p.status === 'on_hold' ? '#92400e' : '#991b1b'
                      }}>{p.status}</span>
                    </td>
                    <td style={{ padding: '6px 12px', textAlign: 'center' }}>{p.progress_pct}%</td>
                    <td style={{ padding: '6px 12px', fontSize: 11, color: '#64748b', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.therapist_notes}</td>
                    <td style={{ padding: '6px 12px', fontSize: 11, color: '#94a3b8' }}>{p.last_updated}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Recently Updated Plans">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Category</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Goal</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Progress</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Sessions</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Updated</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_updates || []).map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 12px' }}>{p.goal_category}</td>
                  <td style={{ padding: '6px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.goal_description}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
                      background: p.status === 'completed' ? '#dcfce7' : p.status === 'active' ? '#dbeafe' : p.status === 'on_hold' ? '#fef3c7' : '#fee2e2',
                      color: p.status === 'completed' ? '#166534' : p.status === 'active' ? '#1e40af' : p.status === 'on_hold' ? '#92400e' : '#991b1b'
                    }}>{p.status}</span>
                  </td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{p.progress_pct}%</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{p.sessions_completed}/{p.sessions_planned}</td>
                  <td style={{ padding: '6px 12px', fontSize: 11, color: '#94a3b8' }}>{p.last_updated}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Goal Categories" span={2}>
        <div style={{ display: 'grid', gap: 10 }}>
          {Object.entries(data.goal_categories || {}).map(([k, v]) => (
            <div key={k} style={{ padding: '10px 14px', background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{k}</div>
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{v}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Plan Statuses">
        <div style={{ display: 'grid', gap: 8 }}>
          {Object.entries(data.statuses || {}).map(([k, v]) => (
            <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
              <span style={{
                display: 'inline-block', padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 600, marginRight: 8,
                background: STATUS_COLORS[k] ? `${STATUS_COLORS[k]}20` : '#f1f5f9',
                color: STATUS_COLORS[k] || '#64748b'
              }}>{k}</span>
              <span style={{ fontSize: 12, color: '#64748b' }}>{v}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Progress Milestones">
        <div style={{ display: 'grid', gap: 8 }}>
          {Object.entries(data.progress_milestones || {}).map(([k, v]) => (
            <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: '#1e293b', marginBottom: 2 }}>{k}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{v}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Clinical Glossary" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {Object.entries(data.glossary || {}).map(([k, v]) => (
            <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: '#1e293b', marginBottom: 2 }}>{k}</div>
              <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.4 }}>{v}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Session Guidelines" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {Object.entries(data.session_guidelines || {}).map(([k, v]) => (
            <div key={k} style={{ padding: '12px 14px', background: '#f0f9ff', borderRadius: 8, textAlign: 'center' }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: '#1e293b', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 12, color: '#3b82f6' }}>{v}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
