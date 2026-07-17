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

function statusColor(s) {
  if (s === 'active') return '#10b981'
  if (s === 'offline') return '#ef4444'
  if (s === 'charging') return '#f59e0b'
  if (s === 'maintenance') return '#8b5cf6'
  return '#64748b'
}

function batteryColor(level) {
  if (level < 20) return '#ef4444'
  if (level < 50) return '#f59e0b'
  return '#10b981'
}

const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#3b82f6', '#ec4899', '#06b6d4', '#f97316']
const STATUS_COLORS = { active: '#10b981', offline: '#ef4444', charging: '#f59e0b', maintenance: '#8b5cf6' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

export default function WearableDevicesDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/wearable-devices/overview`),
      axios.get(`${API_URL}/api/wearable-devices/breakdown`),
      axios.get(`${API_URL}/api/wearable-devices/definitions`),
    ]).then(([o, b, d]) => {
      setOv(o.data); setBd(b.data); setDefs(d.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading wearable devices data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Wearable Devices Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Device fleet analytics — {ov?.total_devices} devices, {ov?.total_patients} patients, avg battery {ov?.avg_battery_level}%
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
      <Card title="Device Fleet Summary" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
          <KPI label="Total Devices" value={ov.total_devices} />
          <KPI label="Active" value={ov.active_count} color="#10b981" />
          <KPI label="Offline" value={ov.offline_count} color="#ef4444" />
          <KPI label="Charging" value={ov.charging_count} color="#f59e0b" />
          <KPI label="Maintenance" value={ov.maintenance_count} color="#8b5cf6" />
          <KPI label="Avg Battery" value={`${ov.avg_battery_level}%`} color={batteryColor(ov.avg_battery_level)} />
          <KPI label="Low Battery" value={ov.low_battery_count} color={ov.low_battery_count > 0 ? '#ef4444' : '#10b981'} />
          <KPI label="Seizure Detection" value={`${ov.seizure_detection_rate}%`} color="#3b82f6" />
          <KPI label="Fall Detection" value={`${ov.fall_detection_rate}%`} color="#06b6d4" />
          <KPI label="Patients" value={ov.total_patients} />
        </div>
      </Card>

      <Card title="Status Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(ov.status_distribution || []).map((s, i) => (
                <Cell key={i} fill={STATUS_COLORS[s.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Device Type Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.device_type_distribution} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" name="Count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Brand Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.brand_distribution} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={150} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" name="Count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Connectivity Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.connectivity_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(ov.connectivity_distribution || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Feature Coverage" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.feature_coverage}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="feature" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 100]} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Bar dataKey="pct" name="Coverage %" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
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
      {(bd.low_battery_devices || []).length > 0 && (
        <Card title="Low Battery Devices" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={th}>Patient</th>
                  <th style={th}>Device</th>
                  <th style={th}>Type</th>
                  <th style={th}>Brand</th>
                  <th style={th}>Battery</th>
                  <th style={th}>Status</th>
                  <th style={th}>Last Sync</th>
                </tr>
              </thead>
              <tbody>
                {(bd.low_battery_devices || []).map((d, i) => (
                  <tr key={i}>
                    <td style={td}>{d.patient_id}</td>
                    <td style={td}>{d.device_id}</td>
                    <td style={td}>{d.device_type}</td>
                    <td style={td}>{d.brand}</td>
                    <td style={td}><Badge text={`${d.battery_level}%`} color={batteryColor(d.battery_level)} /></td>
                    <td style={td}><Badge text={d.status} color={statusColor(d.status)} /></td>
                    <td style={td}>{d.last_sync}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {(bd.offline_devices || []).length > 0 && (
        <Card title="Offline Devices" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={th}>Patient</th>
                  <th style={th}>Device</th>
                  <th style={th}>Type</th>
                  <th style={th}>Brand</th>
                  <th style={th}>Battery</th>
                  <th style={th}>Connectivity</th>
                  <th style={th}>Last Sync</th>
                </tr>
              </thead>
              <tbody>
                {(bd.offline_devices || []).map((d, i) => (
                  <tr key={i}>
                    <td style={td}>{d.patient_id}</td>
                    <td style={td}>{d.device_id}</td>
                    <td style={td}>{d.device_type}</td>
                    <td style={td}>{d.brand}</td>
                    <td style={td}><Badge text={`${d.battery_level}%`} color={batteryColor(d.battery_level)} /></td>
                    <td style={td}>{d.connectivity}</td>
                    <td style={td}>{d.last_sync}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Devices by Brand" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Brand</th>
                <th style={th}>Total</th>
                <th style={th}>Active</th>
                <th style={th}>Avg Battery</th>
              </tr>
            </thead>
            <tbody>
              {(bd.devices_by_brand || []).map((b, i) => (
                <tr key={i}>
                  <td style={td}><strong>{b.brand}</strong></td>
                  <td style={td}>{b.total}</td>
                  <td style={td}><Badge text={b.active} color={b.active > 0 ? '#10b981' : '#ef4444'} /></td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3 }}>
                        <div style={{ width: `${b.avg_battery}%`, height: '100%', background: batteryColor(b.avg_battery), borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{b.avg_battery}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Patient Device Summary">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Patient</th>
                <th style={th}>Devices</th>
                <th style={th}>Avg Battery</th>
                <th style={th}>Offline</th>
              </tr>
            </thead>
            <tbody>
              {(bd.patient_device_summary || []).slice(0, 20).map((p, i) => (
                <tr key={i}>
                  <td style={td}>{p.patient_id}</td>
                  <td style={td}>{p.device_count}</td>
                  <td style={td}><Badge text={`${p.avg_battery}%`} color={batteryColor(p.avg_battery)} /></td>
                  <td style={td}>{p.any_offline ? <Badge text="Yes" color="#ef4444" /> : <Badge text="No" color="#10b981" />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="All Devices" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th}>Patient</th>
                <th style={th}>Device</th>
                <th style={th}>Type</th>
                <th style={th}>Brand</th>
                <th style={th}>Status</th>
                <th style={th}>Battery</th>
                <th style={th}>Connectivity</th>
                <th style={th}>Seizure Det.</th>
                <th style={th}>Fall Det.</th>
                <th style={th}>Last Sync</th>
              </tr>
            </thead>
            <tbody>
              {(bd.all_devices || []).map((d, i) => (
                <tr key={i}>
                  <td style={td}>{d.patient_id}</td>
                  <td style={td}>{d.device_id}</td>
                  <td style={td}>{d.device_type}</td>
                  <td style={td}>{d.brand}</td>
                  <td style={td}><Badge text={d.status} color={statusColor(d.status)} /></td>
                  <td style={td}><Badge text={`${d.battery_level}%`} color={batteryColor(d.battery_level)} /></td>
                  <td style={td}>{d.connectivity}</td>
                  <td style={td}>{d.seizure_detection ? <Badge text="On" color="#10b981" /> : <Badge text="Off" color="#94a3b8" />}</td>
                  <td style={td}>{d.fall_detection ? <Badge text="On" color="#10b981" /> : <Badge text="Off" color="#94a3b8" />}</td>
                  <td style={td}>{d.last_sync}</td>
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
      <Card title="Device Types" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Type</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.device_types || []).map((t, i) => (
              <tr key={i}><td style={td}><strong>{t.type}</strong></td><td style={td}>{t.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Device Statuses">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Status</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.status_descriptions || []).map((s, i) => (
              <tr key={i}><td style={td}><Badge text={s.status} color={statusColor(s.status)} /></td><td style={td}>{s.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Connectivity Types">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Type</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.connectivity_descriptions || []).map((c, i) => (
              <tr key={i}><td style={td}><strong>{c.type}</strong></td><td style={td}>{c.description}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Monitoring Features" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead><tr><th style={th}>Feature</th><th style={th}>Description</th></tr></thead>
          <tbody>
            {(defs.feature_descriptions || []).map((f, i) => (
              <tr key={i}><td style={td}><strong>{f.feature}</strong></td><td style={td}>{f.description}</td></tr>
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
    </div>
  )
}
