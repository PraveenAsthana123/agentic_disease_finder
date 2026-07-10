import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
]

const sevColor = (sev) => {
  if (sev === 'critical') return '#ef4444'
  if (sev === 'high') return '#f97316'
  if (sev === 'medium') return '#f59e0b'
  if (sev === 'low') return '#6366f1'
  return '#8b5cf6'
}

const healthColor = (score) => {
  if (score >= 90) return '#10b981'
  if (score >= 70) return '#f59e0b'
  return '#ef4444'
}

const card = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  marginBottom: 16,
}

const badge = (bg) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: bg,
})

export default function MobileAlertsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    const endpoints = {
      overview: '/api/mobile-alerts/overview',
      rules: '/api/mobile-alerts/breakdown',
      events: '/api/mobile-alerts/breakdown',
      escalation: '/api/mobile-alerts/breakdown',
      definitions: '/api/mobile-alerts/definitions',
    }
    const url = endpoints[tab] || endpoints.overview
    axios.get(API + url)
      .then(r => {
        if (tab === 'overview') setOverview(r.data)
        else if (tab === 'definitions') setDefinitions(r.data)
        else setBreakdown(r.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [tab])

  const tabs = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'rules', label: '⚙️ Alert Rules' },
    { id: 'events', label: '🚨 SOS Events' },
    { id: 'escalation', label: '📞 Escalation' },
    { id: 'definitions', label: '📖 Definitions' },
  ]

  const renderOverview = () => {
    if (!overview) return null
    const s = overview.summary
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
          {[
            { label: 'Health Score', value: s.health_score, color: healthColor(s.health_score), suffix: '%' },
            { label: 'Total Events', value: s.total_events, color: '#6366f1' },
            { label: 'Critical Events', value: s.critical_events, color: '#ef4444' },
            { label: 'Ack Rate', value: s.ack_rate_pct, color: '#10b981', suffix: '%' },
            { label: 'Avg Response', value: s.avg_response_sec, color: '#f59e0b', suffix: 's' },
            { label: 'Active Rules', value: `${s.active_rules}/${s.total_rules}`, color: '#8b5cf6' },
            { label: 'Patients', value: s.patients_monitored, color: '#14b8a6' },
            { label: 'Unresolved', value: s.unresolved, color: s.unresolved > 0 ? '#ef4444' : '#10b981' },
          ].map((m, i) => (
            <div key={i} style={{ ...card, textAlign: 'center' }}>
              <div style={{ fontSize: 12, color: '#6b7280' }}>{m.label}</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: m.color }}>{m.value}{m.suffix || ''}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Severity Distribution</h4>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.severity_distribution} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={80} label={({ severity, pct }) => `${severity} ${pct}%`}>
                  {overview.severity_distribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Hourly Volume (24h)</h4>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={overview.hourly_volume}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="events" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Channel Usage</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.channel_usage}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Response Time Buckets</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.response_time_buckets}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Recent SOS Events</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                {['ID', 'Time', 'Severity', 'Description', 'Patient', 'Response', 'Status'].map(h => (
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#6b7280' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.recent_events.map((ev, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: '8px 6px', fontFamily: 'monospace' }}>{ev.id}</td>
                  <td style={{ padding: '8px 6px' }}>{new Date(ev.timestamp).toLocaleString()}</td>
                  <td style={{ padding: '8px 6px' }}><span style={badge(sevColor(ev.severity))}>{ev.severity}</span></td>
                  <td style={{ padding: '8px 6px' }}>{ev.description}</td>
                  <td style={{ padding: '8px 6px', fontFamily: 'monospace' }}>{ev.patient_id}</td>
                  <td style={{ padding: '8px 6px' }}>{ev.response_time_sec ? `${ev.response_time_sec}s` : '—'}</td>
                  <td style={{ padding: '8px 6px' }}>
                    <span style={badge(ev.resolved ? '#10b981' : '#ef4444')}>{ev.resolved ? 'Resolved' : 'Open'}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const renderRules = () => {
    if (!breakdown) return null
    return (
      <div>
        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Alert Rules Configuration</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                {['Rule', 'Category', 'Severity', 'Trigger', 'Cooldown', 'Channels', 'Fired', 'Ack%', 'Avg RT'].map(h => (
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#6b7280' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.rule_stats.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.name}</td>
                  <td style={{ padding: '8px 6px' }}>{r.category}</td>
                  <td style={{ padding: '8px 6px' }}><span style={badge(sevColor(r.severity))}>{r.severity}</span></td>
                  <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 11 }}>{r.trigger}</td>
                  <td style={{ padding: '8px 6px' }}>{r.cooldown_min}m</td>
                  <td style={{ padding: '8px 6px' }}>{r.channels.join(', ')}</td>
                  <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.times_fired}</td>
                  <td style={{ padding: '8px 6px' }}>{r.ack_rate_pct}%</td>
                  <td style={{ padding: '8px 6px' }}>{r.avg_response_sec}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Times Fired by Rule</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={breakdown.rule_stats} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="times_fired" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Channel Delivery Stats</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={breakdown.channel_stats}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="delivered" fill="#14b8a6" radius={[4, 4, 0, 0]} name="Delivered" />
                <Bar dataKey="acknowledged" fill="#6366f1" radius={[4, 4, 0, 0]} name="Acknowledged" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    )
  }

  const renderEvents = () => {
    if (!breakdown) return null
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Daily Event Trend (14 days)</h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={breakdown.daily_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="events" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Top Patients by Alert Count</h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.top_patients}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="total" fill="#6366f1" radius={[4, 4, 0, 0]} name="Total" />
                <Bar dataKey="critical" fill="#ef4444" radius={[4, 4, 0, 0]} name="Critical" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Full Event Log</h4>
          <div style={{ maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb', position: 'sticky', top: 0, background: '#fff' }}>
                  {['ID', 'Time', 'Severity', 'Rule', 'Patient', 'Channel', 'Escalation', 'Response', 'Status'].map(h => (
                    <th key={h} style={{ padding: '6px 4px', textAlign: 'left', color: '#6b7280' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.all_events.map((ev, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '6px 4px', fontFamily: 'monospace' }}>{ev.id}</td>
                    <td style={{ padding: '6px 4px' }}>{new Date(ev.timestamp).toLocaleString()}</td>
                    <td style={{ padding: '6px 4px' }}><span style={badge(sevColor(ev.severity))}>{ev.severity}</span></td>
                    <td style={{ padding: '6px 4px' }}>{ev.rule_id}</td>
                    <td style={{ padding: '6px 4px', fontFamily: 'monospace' }}>{ev.patient_id}</td>
                    <td style={{ padding: '6px 4px' }}>{ev.channel_used}</td>
                    <td style={{ padding: '6px 4px' }}>{ev.escalation_name}</td>
                    <td style={{ padding: '6px 4px' }}>{ev.response_time_sec ? `${ev.response_time_sec}s` : '—'}</td>
                    <td style={{ padding: '6px 4px' }}>
                      <span style={badge(ev.resolved ? '#10b981' : '#ef4444')}>{ev.resolved ? 'Resolved' : 'Open'}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    )
  }

  const renderEscalation = () => {
    if (!breakdown) return null
    return (
      <div>
        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Escalation Chain</h4>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
            {breakdown.escalation_detail.map((tier, i) => (
              <React.Fragment key={i}>
                <div style={{ ...card, minWidth: 180, textAlign: 'center', border: '2px solid #e5e7eb', marginBottom: 0 }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: COLORS[i] }}>Tier {tier.tier}</div>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>{tier.name}</div>
                  <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 6 }}>{tier.description}</div>
                  <div style={{ fontSize: 12 }}>Delay: <strong>{tier.delay_sec}s</strong></div>
                  <div style={{ fontSize: 12 }}>Events: <strong>{tier.events_reached}</strong> ({tier.pct_of_total}%)</div>
                  <div style={{ fontSize: 11, color: '#6b7280', marginTop: 4 }}>{tier.contacts.join(', ')}</div>
                </div>
                {i < breakdown.escalation_detail.length - 1 && (
                  <div style={{ fontSize: 24, color: '#d1d5db' }}>→</div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Events by Escalation Tier</h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.escalation_detail}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="events_reached" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Escalation Coverage (%)</h4>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={breakdown.escalation_detail} dataKey="events_reached" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, pct_of_total }) => `${name} ${pct_of_total}%`}>
                  {breakdown.escalation_detail.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!definitions) return null
    return (
      <div>
        <div style={card}>
          <h4 style={{ margin: '0 0 16px' }}>Mobile Alerts & SOS Concepts</h4>
          {definitions.concepts.map((c, i) => (
            <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: '1px solid #f3f4f6' }}>
              <div style={{ fontWeight: 700, color: '#1f2937', marginBottom: 4 }}>{c.term}</div>
              <div style={{ fontSize: 13, color: '#4b5563' }}>{c.definition}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Severity Levels</h4>
            {definitions.severity_levels.map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <span style={badge(s.color)}>{s.level}</span>
                <span style={{ fontSize: 13, color: '#4b5563' }}>{s.description}</span>
              </div>
            ))}
          </div>
          <div style={card}>
            <h4 style={{ margin: '0 0 12px' }}>Alert Categories</h4>
            {definitions.alert_categories.map((c, i) => (
              <div key={i} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #f3f4f6' }}>
                <div style={{ fontWeight: 600, color: '#1f2937' }}>{c.category}</div>
                <div style={{ fontSize: 13, color: '#4b5563' }}>{c.description}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 22 }}>Mobile Alerts / SOS Dashboard</h2>
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#6366f1' : '#f3f4f6', color: tab === t.id ? '#fff' : '#374151',
          }}>{t.label}</button>
        ))}
      </div>
      {loading && <div style={card}>Loading...</div>}
      {error && <div style={{ ...card, color: '#ef4444' }}>Error: {error}</div>}
      {!loading && !error && tab === 'overview' && renderOverview()}
      {!loading && !error && tab === 'rules' && renderRules()}
      {!loading && !error && tab === 'events' && renderEvents()}
      {!loading && !error && tab === 'escalation' && renderEscalation()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
