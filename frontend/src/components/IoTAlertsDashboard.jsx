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

function Badge({ text, color }) {
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
      background: color + '20', color: color
    }}>{text}</span>
  )
}

function severityColor(sev) {
  if (sev === 'critical') return '#ef4444'
  if (sev === 'warning') return '#f59e0b'
  return '#3b82f6'
}

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const SEV_COLORS = { critical: '#ef4444', warning: '#f59e0b', info: '#3b82f6' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

export default function IoTAlertsDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/iot-alerts/overview`),
      axios.get(`${API_URL}/api/iot-alerts/breakdown`),
      axios.get(`${API_URL}/api/iot-alerts/definitions`),
    ]).then(([o, b, d]) => {
      setOv(o.data); setBd(b.data); setDefs(d.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading IoT alerts data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>IoT Alerts Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Device alert analytics — {ov?.total_alerts} alerts, {ov?.total_patients} patients, {ov?.total_devices} devices
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t.id ? '#1e293b' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && ov && renderOverview(ov)}
      {tab === 'breakdown' && bd && renderBreakdown(bd)}
      {tab === 'definitions' && defs && renderDefinitions(defs)}
    </div>
  )
}

function renderOverview(ov) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Alert Summary" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
          <KPI label="Total Alerts" value={ov.total_alerts} />
          <KPI label="Critical" value={ov.critical_count} color="#ef4444" />
          <KPI label="Warning" value={ov.warning_count} color="#f59e0b" />
          <KPI label="Info" value={ov.info_count} color="#3b82f6" />
          <KPI label="Unresolved" value={ov.unresolved_count} color="#ef4444" />
          <KPI label="Unacknowledged" value={ov.unacknowledged_count} color="#f59e0b" />
          <KPI label="Ack Rate" value={`${ov.acknowledgment_rate}%`} color={ov.acknowledgment_rate >= 70 ? '#10b981' : '#ef4444'} />
          <KPI label="Resolution Rate" value={`${ov.resolution_rate}%`} color={ov.resolution_rate >= 60 ? '#10b981' : '#ef4444'} />
          <KPI label="Patients" value={ov.total_patients} />
          <KPI label="Devices" value={ov.total_devices} />
        </div>
      </Card>

      <Card title="Severity Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.severity_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(ov.severity_distribution || []).map((s, i) => (
                <Cell key={i} fill={SEV_COLORS[s.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Alert Type Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.alert_type_distribution} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={130} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" name="Count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Alert Trend" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={ov.monthly_trend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="critical" stroke="#ef4444" name="Critical" strokeWidth={2} />
            <Line type="monotone" dataKey="warning" stroke="#f59e0b" name="Warning" strokeWidth={2} />
            <Line type="monotone" dataKey="info" stroke="#3b82f6" name="Info" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderBreakdown(bd) {
  const th = { padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }
  const td = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #e2e8f0' }
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Unresolved Alerts" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#fef2f2' }}>
                <th style={th}>Patient</th>
                <th style={th}>Alert Type</th>
                <th style={th}>Severity</th>
                <th style={th}>Device</th>
                <th style={th}>Acknowledged</th>
                <th style={th}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {(bd.unresolved_alerts || []).map((a, i) => (
                <tr key={i}>
                  <td style={td}>{a.patient_id || '—'}</td>
                  <td style={td}>{a.alert_type}</td>
                  <td style={td}><Badge text={a.severity} color={severityColor(a.severity)} /></td>
                  <td style={td}>{a.device_id || a.gateway_id || '—'}</td>
                  <td style={td}>{a.acknowledged ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>{a.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Alerts by Device" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Device</th>
                <th style={th}>Total</th>
                <th style={th}>Critical</th>
                <th style={th}>Unresolved</th>
              </tr>
            </thead>
            <tbody>
              {(bd.alerts_by_device || []).map((d, i) => (
                <tr key={i}>
                  <td style={td}>{d.device_id}</td>
                  <td style={td}>{d.total}</td>
                  <td style={td}><Badge text={d.critical} color={d.critical > 0 ? '#ef4444' : '#10b981'} /></td>
                  <td style={td}><Badge text={d.unresolved} color={d.unresolved > 0 ? '#f59e0b' : '#10b981'} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Patient Alert Summary">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Patient</th>
                <th style={th}>Total</th>
                <th style={th}>Critical</th>
                <th style={th}>Resolved %</th>
              </tr>
            </thead>
            <tbody>
              {(bd.patient_alert_summary || []).map((p, i) => (
                <tr key={i}>
                  <td style={td}>{p.patient_id}</td>
                  <td style={td}>{p.total}</td>
                  <td style={td}><Badge text={p.critical} color={p.critical > 0 ? '#ef4444' : '#10b981'} /></td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3 }}>
                        <div style={{ width: `${p.resolved_pct}%`, height: '100%', background: p.resolved_pct >= 75 ? '#10b981' : p.resolved_pct >= 50 ? '#f59e0b' : '#ef4444', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{p.resolved_pct}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Recent Alerts" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Patient</th>
                <th style={th}>Alert Type</th>
                <th style={th}>Severity</th>
                <th style={th}>Device</th>
                <th style={th}>Acknowledged</th>
                <th style={th}>Resolved</th>
                <th style={th}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {(bd.recent_alerts || []).map((a, i) => (
                <tr key={i}>
                  <td style={td}>{a.patient_id || '—'}</td>
                  <td style={td}>{a.alert_type}</td>
                  <td style={td}><Badge text={a.severity} color={severityColor(a.severity)} /></td>
                  <td style={td}>{a.device_id || a.gateway_id || '—'}</td>
                  <td style={td}>{a.acknowledged ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>{a.resolved ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#ef4444" />}</td>
                  <td style={td}>{a.timestamp}</td>
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
      <Card title="Alert Types" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Type</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.alert_types || []).map((t, i) => (
              <tr key={i}><td style={td}><strong>{t.type}</strong></td><td style={td}>{t.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Severity Levels">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Level</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.severity_levels || []).map((s, i) => (
              <tr key={i}><td style={td}><Badge text={s.level} color={severityColor(s.level)} /></td><td style={td}>{s.description}</td></tr>
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

      <Card title="Clinical Notes">
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
          {(defs.clinical_notes || []).map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      </Card>

      {defs.data_source && (
        <Card title="Data Source">
          <p style={{ fontSize: 12, color: '#475569', margin: 0 }}>{defs.data_source}</p>
        </Card>
      )}
    </div>
  )
}
