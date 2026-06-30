import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']
const HEALTH_COLORS = { healthy: '#10b981', degraded: '#f59e0b', critical: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function EventQueueDashboard() {
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
          axios.get(`${API_URL}/event-queue/overview`),
          axios.get(`${API_URL}/event-queue/breakdown`),
          axios.get(`${API_URL}/event-queue/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load event queue data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9881;</div>
      Loading event queue data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Event queue data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const healthColor = HEALTH_COLORS[s.health_status] || '#64748b'
  const actionData = (overview.action_distribution || []).slice(0, 10)
  const queueData = (overview.component_queues || []).slice(0, 12)
  const actorData = overview.actor_distribution || []
  const dailyData = overview.daily_volume || []
  const hourlyData = overview.hourly_distribution || []
  const msgQueue = overview.message_queue || []
  const uploadQueue = overview.upload_queue || []
  const crossTab = (breakdown?.cross_tab || []).slice(0, 30)
  const recentEvents = breakdown?.recent_events || []
  const patientEvents = breakdown?.patient_events || []
  const queueStats = breakdown?.queue_stats || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#64748b'
  })
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  const renderOverview = () => (
    <>
      {/* Health banner */}
      <div style={{ ...cardStyle, background: `${healthColor}11`, border: `2px solid ${healthColor}44` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Queue Health</div>
            <div style={{ fontSize: 36, fontWeight: 800, color: healthColor, textTransform: 'uppercase' }}>{s.health_status}</div>
          </div>
          <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Throughput (24h)</span><br/><strong>{fmt(s.throughput_24h)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Throughput (1h)</span><br/><strong>{fmt(s.throughput_1h)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Blocked</span><br/><strong>{fmt(s.blocked_events)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Errors</span><br/><strong>{fmt(s.error_events)}</strong></div>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
        {[
          { label: 'Total Events', value: s.total_events, color: '#3b82f6' },
          { label: 'Queued Items', value: s.total_queued_items, color: '#10b981' },
          { label: 'Conversations', value: s.total_conversations, color: '#8b5cf6' },
          { label: 'Assessments', value: s.total_assessments, color: '#f59e0b' },
          { label: 'Uploads', value: s.total_uploads, color: '#06b6d4' },
          { label: 'Distinct Queues', value: s.distinct_queues, color: '#ec4899' },
          { label: 'Event Types', value: s.distinct_event_types, color: '#64748b' },
          { label: 'Patients', value: s.unique_patients, color: '#84cc16' },
        ].map((k, i) => (
          <div key={i} style={kpiStyle(k.color)}>
            <div style={{ fontSize: 11, color: '#64748b' }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>{fmt(k.value)}</div>
          </div>
        ))}
      </div>

      {/* Daily volume trend */}
      {dailyData.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Daily Event Volume (14 days)</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={dailyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="events" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Action distribution + Actor distribution side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        {actionData.length > 0 && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Event Type Distribution</div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={actionData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="action" type="category" tick={{ fontSize: 10 }} width={100} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {actorData.length > 0 && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Actor Distribution</div>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={actorData} dataKey="events" nameKey="actor" cx="50%" cy="50%" outerRadius={80} label={({ actor, percent }) => `${actor} ${(percent * 100).toFixed(0)}%`}>
                  {actorData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Hourly distribution */}
      {hourlyData.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Hourly Event Distribution (UTC)</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="events" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )

  const renderQueues = () => (
    <>
      {/* Component queue stats table */}
      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Queue Stats by Component</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Queue', 'Events', 'Action Types', 'Patients', 'First Event', 'Last Event'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {queueStats.map((q, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{q.queue}</td>
                  <td style={{ padding: '6px 10px' }}>{fmt(q.total_events)}</td>
                  <td style={{ padding: '6px 10px' }}>{q.action_types}</td>
                  <td style={{ padding: '6px 10px' }}>{q.patients}</td>
                  <td style={{ padding: '6px 10px', fontSize: 10, color: '#64748b' }}>{q.first_event?.slice(0, 16)}</td>
                  <td style={{ padding: '6px 10px', fontSize: 10, color: '#64748b' }}>{q.last_event?.slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Component queue bar chart */}
      {queueData.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Events per Queue</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={queueData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis dataKey="queue" type="category" tick={{ fontSize: 10 }} width={120} />
              <Tooltip />
              <Bar dataKey="events" fill="#10b981" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Message queue + Upload queue */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        {msgQueue.length > 0 && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Message Queue (by role)</div>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={msgQueue} dataKey="messages" nameKey="role" cx="50%" cy="50%" outerRadius={70} label={({ role, percent }) => `${role} ${(percent * 100).toFixed(0)}%`}>
                  {msgQueue.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
        {uploadQueue.length > 0 && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Upload Queue (by status)</div>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={uploadQueue} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={70} label={({ status, percent }) => `${status} ${(percent * 100).toFixed(0)}%`}>
                  {uploadQueue.map((_, i) => <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Patient event counts */}
      {patientEvents.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Events per Patient</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={patientEvents}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="events" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )

  const renderEvents = () => (
    <>
      {/* Cross-tab */}
      {crossTab.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Component x Action Cross-Tab (top 30)</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Component', 'Action', 'Count'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {crossTab.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.component}</td>
                    <td style={{ padding: '6px 10px' }}>{r.action}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(r.count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent events */}
      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Recent Events (last 50)</div>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                {['ID', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                  <th key={h} style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recentEvents.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px 8px', color: '#64748b' }}>{e.id}</td>
                  <td style={{ padding: '4px 8px' }}>{e.patient_id || '--'}</td>
                  <td style={{ padding: '4px 8px', fontWeight: 600 }}>{e.component}</td>
                  <td style={{ padding: '4px 8px' }}>
                    <span style={{ background: '#eff6ff', color: '#3b82f6', padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 600 }}>{e.action}</span>
                  </td>
                  <td style={{ padding: '4px 8px' }}>{e.actor}</td>
                  <td style={{ padding: '4px 8px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || '--'}</td>
                  <td style={{ padding: '4px 8px', fontSize: 10, color: '#64748b' }}>{e.ts_utc?.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )

  const renderDefinitions = () => (
    <div style={cardStyle}>
      <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>Metric Definitions</div>
      {(defs?.definitions || []).map((d, i) => (
        <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < (defs?.definitions || []).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
          <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.term}</div>
          <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{d.definition}</div>
        </div>
      ))}
    </div>
  )

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 18 }}>
        <h2 style={{ fontSize: 20, fontWeight: 800, margin: 0 }}>Event / Queue Dashboard</h2>
        <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0 0' }}>
          Real event-processing pipeline from clinical.db &mdash; {fmt(s.total_events)} events across {fmt(s.distinct_queues)} queues
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'queues', label: 'Queues' },
          { id: 'events', label: 'Event Log' },
          { id: 'definitions', label: 'Definitions' },
        ].map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'queues' && renderQueues()}
      {tab === 'events' && renderEvents()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
