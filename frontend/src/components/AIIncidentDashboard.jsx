import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend
} from 'recharts'

const API_URL = '/api'

const SEV_COLORS = {
  Critical: '#ef4444',
  High: '#f97316',
  Medium: '#eab308',
  Low: '#3b82f6'
}
const STATUS_COLORS = {
  open: '#ef4444',
  investigating: '#f97316',
  mitigated: '#eab308',
  resolved: '#22c55e',
  closed: '#94a3b8'
}
const CAT_COLORS = ['#3b82f6','#ef4444','#22c55e','#f97316','#8b5cf6','#06b6d4','#ec4899','#14b8a6']

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

function SeverityBadge({ severity }) {
  const color = SEV_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{severity || 'Unknown'}</span>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status || 'unknown'}</span>
  )
}

export default function AIIncidentDashboard() {
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
          axios.get(`${API_URL}/ai-incident/overview`),
          axios.get(`${API_URL}/ai-incident/breakdown`),
          axios.get(`${API_URL}/ai-incident/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI Incident data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No AI incident data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'incidents', label: 'Incident Log' },
    { id: 'analysis', label: 'Root Cause Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const sevData = overview?.severity_distribution || []
  const catData = (overview?.category_distribution || []).map((d, i) => ({ ...d, color: CAT_COLORS[i % CAT_COLORS.length] }))
  const timelineData = overview?.incident_timeline || []
  const recentIncidents = breakdown?.recent_incidents || []
  const bySource = breakdown?.by_source || []
  const rootCauses = breakdown?.root_cause_analysis || []
  const patientImpact = breakdown?.patient_impact || []
  const responders = breakdown?.responder_workload || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Incident Management Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          AI system incident tracking, severity analysis, MTTR metrics, and root cause investigation
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
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Incidents" value={fmt(overview.kpis?.total_incidents)} />
              <KPI label="Open" value={fmt(overview.kpis?.open_incidents)} color="#ef4444" />
              <KPI label="Resolved" value={fmt(overview.kpis?.resolved_incidents)} color="#22c55e" />
              <KPI label="MTTR (hrs)" value={fmt(overview.kpis?.mttr_hours)} color="#3b82f6" />
              <KPI label="Critical" value={fmt(overview.kpis?.severity_critical)} color="#ef4444" />
              <KPI label="Resolution Rate" value={overview.resolution_rate_pct != null ? fmt(overview.resolution_rate_pct) + '%' : '--'} color="#22c55e" />
            </div>
          </Card>

          {/* Severity Distribution */}
          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sevData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Incidents">
                  {sevData.map((d, i) => <Cell key={i} fill={SEV_COLORS[d.name] || '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Category Distribution */}
          <Card title="Incident Categories" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={catData} dataKey="count" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {catData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {catData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.count}
                </span>
              ))}
            </div>
          </Card>

          {/* Incident Timeline */}
          <Card title="Incident Timeline (30 days)" span={3}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="incidents" stroke="#ef4444" name="Incidents" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="resolved" stroke="#22c55e" name="Resolved" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Incident Log */}
      {tab === 'incidents' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Recent Incidents">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>ID</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Timestamp</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Severity</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Resolution (hrs)</th>
                  </tr>
                </thead>
                <tbody>
                  {recentIncidents.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 11, color: '#3b82f6' }}>INC-{row.id || i + 1}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.timestamp || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.category || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><SeverityBadge severity={row.severity} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.description || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.patient_id || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, fontWeight: 600, color: '#475569' }}>{row.resolution_time_hrs != null ? fmt(row.resolution_time_hrs) : '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Patient Impact */}
          <Card title="Patient Impact Summary">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Incidents</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Most Recent</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Max Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {patientImpact.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.patient_id || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.incident_count)}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.most_recent || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><SeverityBadge severity={row.severity_max} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Root Cause Analysis */}
      {tab === 'analysis' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Root Cause Bar */}
          <Card title="Root Cause Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={rootCauses}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="root_cause" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" name="Incidents" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* By Source */}
          <Card title="Incidents by Source">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={bySource.map((d, i) => ({ ...d, color: CAT_COLORS[i % CAT_COLORS.length] }))}
                  dataKey="count" nameKey="source" cx="50%" cy="50%" innerRadius={35} outerRadius={70} paddingAngle={2}>
                  {bySource.map((d, i) => <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {bySource.map((d, i) => (
                <span key={d.source} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: CAT_COLORS[i % CAT_COLORS.length], marginRight: 4 }} />
                  {d.source}: {d.count}
                </span>
              ))}
            </div>
          </Card>

          {/* Responder Workload */}
          <Card title="Responder Workload">
            <div style={{ maxHeight: 250, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Responder</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Handled</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Resolution (hrs)</th>
                  </tr>
                </thead>
                <tbody>
                  {responders.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.responder || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.incidents_handled)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{fmt(row.avg_resolution_hrs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 4: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Severity Levels */}
          {defs.severity_levels && (
            <Card title="Severity Levels">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                {defs.severity_levels.map(s => (
                  <div key={s.level} style={{
                    padding: 12, borderRadius: 8, border: '1px solid #e2e8f0',
                    borderLeft: `4px solid ${SEV_COLORS[s.level] || '#94a3b8'}`
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 13, color: SEV_COLORS[s.level] || '#334155' }}>{s.level}</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{s.description}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Incident Categories */}
          {defs.incident_categories && (
            <Card title="Incident Categories">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                {defs.incident_categories.map((c, i) => (
                  <div key={c.name} style={{ padding: 10, borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                    <div style={{ fontWeight: 600, fontSize: 12, color: '#334155' }}>{c.name}</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{c.description}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Metrics */}
          {defs.metrics && (
            <Card title="Key Metrics">
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Metric</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.metrics.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{m.name}</td>
                      <td style={{ padding: '6px 8px', color: '#475569' }}>{m.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Methodology */}
          {defs.methodology && (
            <Card title="Methodology">
              <p style={{ fontSize: 12, color: '#475569', lineHeight: 1.6, margin: 0 }}>{defs.methodology}</p>
            </Card>
          )}

          {/* References */}
          {defs.references && (
            <Card title="Standards & References">
              <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
                {defs.references.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
