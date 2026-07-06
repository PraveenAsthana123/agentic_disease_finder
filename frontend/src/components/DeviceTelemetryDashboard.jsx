import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

function Badge({ text, bg, fg }) {
  return (
    <span style={{
      background: bg, color: fg, padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600
    }}>{text}</span>
  )
}

function SeverityBadge({ severity }) {
  const map = {
    critical: { bg: '#fee2e2', fg: '#991b1b' },
    warning: { bg: '#fef3c7', fg: '#92400e' },
    info: { bg: '#dbeafe', fg: '#1e40af' },
  }
  const s = map[severity] || { bg: '#f1f5f9', fg: '#475569' }
  return <Badge text={severity} bg={s.bg} fg={s.fg} />
}

function StatusBadge({ status }) {
  const map = {
    online: { bg: '#dcfce7', fg: '#166534' },
    active: { bg: '#dcfce7', fg: '#166534' },
    offline: { bg: '#fee2e2', fg: '#991b1b' },
    degraded: { bg: '#fef3c7', fg: '#92400e' },
  }
  const s = map[status] || { bg: '#f1f5f9', fg: '#475569' }
  return <Badge text={status} bg={s.bg} fg={s.fg} />
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v <= 1 ? (v * 100).toFixed(1) + '%' : v.toFixed(1) + '%') : String(v)
}

const TABS = ['Overview', 'Battery & Signal', 'Alerts', 'Devices', 'Reference']

export default function DeviceTelemetryDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/device-telemetry/overview`),
      axios.get(`${API}/device-telemetry/breakdown`),
      axios.get(`${API}/device-telemetry/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading device telemetry data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const kpis = ov?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px' }}>
        Device Telemetry Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 20px' }}>
        Fleet health, battery monitoring, signal quality, and alert audit
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#1e293b' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: 600, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} ov={ov} />}
      {tab === 1 && <BatterySignalTab bd={bd} ov={ov} />}
      {tab === 2 && <AlertsTab bd={bd} ov={ov} />}
      {tab === 3 && <DevicesTab bd={bd} />}
      {tab === 4 && <ReferenceTab df={df} />}
    </div>
  )
}

/* ── Tab 1: Overview ── */
function OverviewTab({ kpis, ov }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Fleet KPIs" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <KPI label="Total Devices" value={fmt(kpis.total_devices)} color="#3b82f6" />
          <KPI label="Online %" value={fmtPct(kpis.online_pct)} color="#10b981" />
          <KPI label="Avg Battery %" value={fmtPct(kpis.avg_battery_pct)} color="#f59e0b" />
          <KPI label="Unresolved Alerts" value={fmt(kpis.unresolved_alerts)} color="#ef4444" />
        </div>
      </Card>

      <Card title="Battery Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={ov?.battery_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Device Type Breakdown" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={ov?.device_type_breakdown || []} dataKey="count" nameKey="type"
              cx="50%" cy="50%" outerRadius={90} label={({ type, count }) => `${type}: ${count}`}>
              {(ov?.device_type_breakdown || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Fleet Summary" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, padding: '12px 0' }}>
          <KPI label="IoT Sensors" value={fmt(kpis.iot_count)} color="#06b6d4" />
          <KPI label="Wearables" value={fmt(kpis.wearable_count)} color="#8b5cf6" />
          <KPI label="Gateways" value={fmt(kpis.gateway_count)} color="#ec4899" />
          <KPI label="Avg Latency (ms)" value={fmt(kpis.avg_latency_ms)} color="#64748b" />
        </div>
      </Card>
    </div>
  )
}

/* ── Tab 2: Battery & Signal ── */
function BatterySignalTab({ bd, ov }) {
  const kpis = ov?.kpis || {}
  const devices = bd?.per_device || []
  const sortedByBattery = [...devices].sort((a, b) => (a.battery_pct ?? 100) - (b.battery_pct ?? 100))

  function batteryColor(v) {
    if (v == null) return '#94a3b8'
    if (v < 20) return '#ef4444'
    if (v < 50) return '#f59e0b'
    return '#10b981'
  }

  const batteryData = sortedByBattery.map(d => ({
    device: d.device_id,
    battery: d.battery_pct,
    fill: batteryColor(d.battery_pct)
  }))

  const signalData = devices.map(d => ({
    device: d.device_id,
    signal: d.signal_strength
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Battery & Signal KPIs" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <KPI label="Avg Battery" value={fmtPct(kpis.avg_battery_pct)} color="#f59e0b" />
          <KPI label="Low Battery Count" value={fmt(kpis.low_battery_count)} color="#ef4444" />
          <KPI label="Avg Signal Strength" value={fmt(kpis.avg_signal_strength)} color="#3b82f6" sub="dBm" />
        </div>
      </Card>

      <Card title="Per-Device Battery Levels (sorted worst-first)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={batteryData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="device" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
            <YAxis domain={[0, 100]} label={{ value: 'Battery %', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <Tooltip />
            {batteryData.map((d, i) => null)}
            <Bar dataKey="battery" radius={[4, 4, 0, 0]} name="Battery %">
              {batteryData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Signal Strength by Device" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={signalData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="device" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
            <YAxis label={{ value: 'Signal (dBm)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <Tooltip />
            <Bar dataKey="signal" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Signal Strength" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Threshold Reference" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Metric', 'Good', 'Warning', 'Critical'].map(h =>
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: '8px 10px', fontWeight: 600 }}>Battery</td>
              <td style={{ padding: '8px 10px' }}><Badge text="> 50%" bg="#dcfce7" fg="#166534" /></td>
              <td style={{ padding: '8px 10px' }}><Badge text="20-50%" bg="#fef3c7" fg="#92400e" /></td>
              <td style={{ padding: '8px 10px' }}><Badge text="< 20%" bg="#fee2e2" fg="#991b1b" /></td>
            </tr>
            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: '8px 10px', fontWeight: 600 }}>Signal</td>
              <td style={{ padding: '8px 10px' }}><Badge text="> -60 dBm" bg="#dcfce7" fg="#166534" /></td>
              <td style={{ padding: '8px 10px' }}><Badge text="-60 to -80 dBm" bg="#fef3c7" fg="#92400e" /></td>
              <td style={{ padding: '8px 10px' }}><Badge text="< -80 dBm" bg="#fee2e2" fg="#991b1b" /></td>
            </tr>
          </tbody>
        </table>
      </Card>
    </div>
  )
}

/* ── Tab 3: Alerts ── */
function AlertsTab({ bd, ov }) {
  const alerts = bd?.alerts || {}
  const alertList = bd?.recent_alerts || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Alert KPIs" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <KPI label="Total Alerts" value={fmt(alerts.total)} color="#3b82f6" />
          <KPI label="Unresolved" value={fmt(alerts.unresolved)} color="#ef4444" />
          <KPI label="Critical Count" value={fmt(alerts.critical_count)} color="#f59e0b" />
        </div>
      </Card>

      <Card title="Alerts by Type">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={alerts.by_type || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Severity Breakdown">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={alerts.by_severity || []} dataKey="count" nameKey="severity"
              cx="50%" cy="50%" outerRadius={90}
              label={({ severity, count }) => `${severity}: ${count}`}>
              {(alerts.by_severity || []).map((_, i) => {
                const severityColors = { critical: '#ef4444', warning: '#f59e0b', info: '#3b82f6' }
                const entry = (alerts.by_severity || [])[i]
                return <Cell key={i} fill={severityColors[entry?.severity] || COLORS[i % COLORS.length]} />
              })}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title={`Recent Alerts (${alertList.length})`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Date', 'Device', 'Type', 'Severity', 'Message', 'Status'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {alertList.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>{a.date?.split('T')[0] || '--'}</td>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{a.device_id}</td>
                  <td style={{ padding: '8px 10px' }}>{a.type}</td>
                  <td style={{ padding: '8px 10px' }}><SeverityBadge severity={a.severity} /></td>
                  <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis' }}>{a.message || '--'}</td>
                  <td style={{ padding: '8px 10px' }}>{a.resolved ? <Badge text="resolved" bg="#dcfce7" fg="#166534" /> : <Badge text="open" bg="#fee2e2" fg="#991b1b" />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ── Tab 4: Devices ── */
function DevicesTab({ bd }) {
  const devices = bd?.per_device || []
  const wearables = bd?.wearable_devices || []
  const kpis = bd?.device_counts || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Device Type Counts">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <KPI label="IoT Devices" value={fmt(kpis.iot)} color="#06b6d4" />
          <KPI label="Wearable Devices" value={fmt(kpis.wearable)} color="#8b5cf6" />
          <KPI label="Gateways" value={fmt(kpis.gateway)} color="#ec4899" />
        </div>
      </Card>

      <Card title={`All Devices (${devices.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Device ID', 'Type', 'Patient', 'Battery', 'Signal', 'Status', 'Latency', 'Last Seen', 'Firmware'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {devices.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{d.device_id}</td>
                  <td style={{ padding: '8px 10px' }}>{d.type}</td>
                  <td style={{ padding: '8px 10px' }}>{d.patient_id || '--'}</td>
                  <td style={{ padding: '8px 10px' }}>{d.battery_pct != null ? `${d.battery_pct}%` : '--'}</td>
                  <td style={{ padding: '8px 10px' }}>{d.signal_strength != null ? `${d.signal_strength} dBm` : '--'}</td>
                  <td style={{ padding: '8px 10px' }}><StatusBadge status={d.status} /></td>
                  <td style={{ padding: '8px 10px' }}>{d.latency_ms != null ? `${d.latency_ms}ms` : '--'}</td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>{d.last_seen?.split('T')[0] || '--'}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12 }}>{d.firmware || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title={`Wearable Devices (${wearables.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Device ID', 'Brand', 'Type', 'Battery', 'Connectivity', 'Status', 'Last Sync'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {wearables.map((w, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{w.device_id}</td>
                  <td style={{ padding: '8px 10px' }}>{w.brand || '--'}</td>
                  <td style={{ padding: '8px 10px' }}>{w.type || '--'}</td>
                  <td style={{ padding: '8px 10px' }}>{w.battery_pct != null ? `${w.battery_pct}%` : '--'}</td>
                  <td style={{ padding: '8px 10px' }}>{w.connectivity || '--'}</td>
                  <td style={{ padding: '8px 10px' }}><StatusBadge status={w.status} /></td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>{w.last_sync?.split('T')[0] || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ── Tab 5: Reference ── */
function ReferenceTab({ df }) {
  if (!df) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Signal Strength Thresholds">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Range', 'Quality', 'Description'].map(h =>
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {(df.signal_thresholds || []).map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>{s.range}</td>
                <td style={{ padding: '8px 10px' }}>{s.quality}</td>
                <td style={{ padding: '8px 10px', color: '#475569' }}>{s.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Battery Level Thresholds">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Range', 'Status', 'Action'].map(h =>
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {(df.battery_thresholds || []).map((b, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>{b.range}</td>
                <td style={{ padding: '8px 10px' }}>{b.status}</td>
                <td style={{ padding: '8px 10px', color: '#475569' }}>{b.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Alert Severity Definitions">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {(df.severity_definitions || []).map((s, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ marginBottom: 6 }}><SeverityBadge severity={s.severity} /></div>
              <div style={{ fontSize: 12, color: '#475569' }}>{s.description}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Device Type Glossary">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {(df.device_glossary || []).map((g, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b' }}>{g.term}</div>
              <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>{g.definition}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Clinical Importance Notes">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {(df.clinical_notes || []).map((n, i) => (
            <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{n}</li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
