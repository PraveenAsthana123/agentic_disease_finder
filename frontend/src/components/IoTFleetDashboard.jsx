import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b', '#795548', '#9e9e9e']
const SEV_COLORS = { critical: '#f44336', warning: '#ff9800', info: '#1e88e5' }
const STATUS_COLORS = { online: '#4caf50', offline: '#f44336', degraded: '#ff9800', unknown: '#9e9e9e' }

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, warn }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 16px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#f44336' : '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const bg = STATUS_COLORS[status] || '#9e9e9e'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {status || 'unknown'}
    </span>
  )
}

function SevBadge({ sev }) {
  const bg = SEV_COLORS[sev] || '#9e9e9e'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {sev}
    </span>
  )
}

function pct(v) { return v != null ? `${(v * 100).toFixed(1)}%` : '–' }
function num(v, dec = 1) { return v != null ? Number(v).toFixed(dec) : '–' }

export default function IoTFleetDashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const tabs = ['overview', 'devices', 'gateways', 'alerts', 'patients', 'definitions']

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/iot-fleet/overview`),
      axios.get(`${API_URL}/api/iot-fleet/breakdown`),
      axios.get(`${API_URL}/api/iot-fleet/definitions`),
    ])
      .then(([ov, bd, df]) => setData({ overview: ov.data, breakdown: bd.data, definitions: df.data }))
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ color: '#f44336', padding: 24 }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 24, color: '#64748b' }}>Loading IoT Fleet...</div>
  const { overview: ov, breakdown: bd, definitions: df } = data
  const k = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        IoT Fleet Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Device health, gateway uptime, and alert pipeline — real data from iot_devices ({k.total_devices ?? 0} devices),
        iot_gateways ({k.total_gateways ?? 0} gateways), iot_alerts ({k.total_alerts ?? 0} alerts)
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t ? '#1e293b' : '#f1f5f9', color: tab === t ? '#fff' : '#475569'
          }}>
            {t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <>
          <Card title="Fleet KPIs — Devices">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
              <KPI label="Total Devices" value={k.total_devices ?? 0} />
              <KPI label="Online" value={k.online_devices ?? 0} sub={pct(k.device_availability_rate)} />
              <KPI label="Avg Battery %" value={num(k.avg_battery_pct)} warn={k.avg_battery_pct < 30} />
              <KPI label="Low Battery" value={k.low_battery_devices ?? 0} warn={k.low_battery_devices > 0} />
              <KPI label="Avg Latency ms" value={num(k.avg_latency_ms)} />
              <KPI label="Avg Signal dBm" value={num(k.avg_signal_dbm)} />
              <KPI label="Patients Covered" value={k.unique_patients_covered ?? 0} />
            </div>
          </Card>

          <Card title="Fleet KPIs — Gateways & Alerts">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
              <KPI label="Gateways" value={k.total_gateways ?? 0} />
              <KPI label="Gateways Online" value={k.online_gateways ?? 0} />
              <KPI label="Avg Uptime %" value={num(k.avg_gateway_uptime_pct)} />
              <KPI label="Connected Devices" value={k.total_connected_devices ?? 0} />
              <KPI label="Total Alerts" value={k.total_alerts ?? 0} />
              <KPI label="Unresolved" value={k.unresolved_alerts ?? 0} warn={k.unresolved_alerts > 0} />
              <KPI label="Critical Open" value={k.critical_unresolved ?? 0} warn={k.critical_unresolved > 0} />
              <KPI label="Unacknowledged" value={k.unacknowledged_alerts ?? 0} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Device Status Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={ov.device_status_distribution} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, pct: p }) => `${name} ${p ? (p * 100).toFixed(0) : ''}%`}>
                    {(ov.device_status_distribution || []).map((_, i) => (
                      <Cell key={i} fill={STATUS_COLORS[_.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Device Type Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={ov.device_type_distribution} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Alert Severity Distribution">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={ov.alert_severity_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {(ov.alert_severity_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={SEV_COLORS[entry.name] || COLORS[i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Devices by Location">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={ov.location_distribution} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#7c4dff" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── DEVICES TAB ── */}
      {tab === 'devices' && (
        <>
          {(bd.low_battery_devices || []).length > 0 && (
            <Card title={`Low-Battery Devices (< 30%) — ${bd.low_battery_devices.length} devices`}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#fef3c7' }}>
                    {['Device ID', 'Patient', 'Type', 'Location', 'Battery %'].map(h => (
                      <th key={h} style={{ padding: '6px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bd.low_battery_devices.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#fafafa' : '#fff' }}>
                      <td style={{ padding: '5px 10px', fontWeight: 600 }}>{d.device_id}</td>
                      <td style={{ padding: '5px 10px' }}>{d.patient_id ?? '–'}</td>
                      <td style={{ padding: '5px 10px' }}>{d.type}</td>
                      <td style={{ padding: '5px 10px' }}>{d.location}</td>
                      <td style={{ padding: '5px 10px', color: '#f44336', fontWeight: 700 }}>{num(d.battery_pct)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          <Card title={`All Devices (${(bd.device_table || []).length} shown)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Device ID', 'Type', 'Status', 'Patient', 'Location', 'Battery %', 'Signal dBm', 'Latency ms', 'Firmware', 'Last Seen'].map(h => (
                      <th key={h} style={{ padding: '6px 10px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.device_table || []).map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#fafafa' : '#fff' }}>
                      <td style={{ padding: '5px 10px', fontWeight: 600 }}>{d.device_id}</td>
                      <td style={{ padding: '5px 10px' }}>{d.type}</td>
                      <td style={{ padding: '5px 10px' }}><StatusBadge status={d.status} /></td>
                      <td style={{ padding: '5px 10px' }}>{d.patient_id ?? '–'}</td>
                      <td style={{ padding: '5px 10px' }}>{d.location}</td>
                      <td style={{ padding: '5px 10px', color: _safe_float(d.battery_pct) < 20 ? '#f44336' : '#1e293b' }}>{num(d.battery_pct)}%</td>
                      <td style={{ padding: '5px 10px' }}>{num(d.signal_dbm)}</td>
                      <td style={{ padding: '5px 10px' }}>{num(d.latency_ms)}</td>
                      <td style={{ padding: '5px 10px', fontFamily: 'monospace' }}>{d.firmware}</td>
                      <td style={{ padding: '5px 10px', color: '#64748b' }}>{d.last_seen ? d.last_seen.replace('T', ' ').slice(0, 16) : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── GATEWAYS TAB ── */}
      {tab === 'gateways' && (
        <Card title={`Gateway Fleet (${(bd.gateway_table || []).length} gateways)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Gateway ID', 'Location', 'Status', 'Uptime %', 'Connected Devices', 'Firmware', 'Last Heartbeat'].map(h => (
                    <th key={h} style={{ padding: '6px 12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.gateway_table || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#fafafa' : '#fff' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600 }}>{g.gateway_id}</td>
                    <td style={{ padding: '6px 12px' }}>{g.location}</td>
                    <td style={{ padding: '6px 12px' }}><StatusBadge status={g.status} /></td>
                    <td style={{ padding: '6px 12px', color: _safe_float(g.uptime_pct) < 90 ? '#f44336' : '#4caf50', fontWeight: 600 }}>{num(g.uptime_pct)}%</td>
                    <td style={{ padding: '6px 12px', textAlign: 'center' }}>{g.connected_devices}</td>
                    <td style={{ padding: '6px 12px', fontFamily: 'monospace' }}>{g.firmware}</td>
                    <td style={{ padding: '6px 12px', color: '#64748b' }}>{g.last_heartbeat ? g.last_heartbeat.replace('T', ' ').slice(0, 16) : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── ALERTS TAB ── */}
      {tab === 'alerts' && (
        <>
          <Card title="Alert Type Breakdown">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bd.alert_type_breakdown} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="type" width={160} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="total" name="Total" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                <Bar dataKey="unresolved" name="Unresolved" fill="#f44336" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title={`Unresolved Alerts (${(bd.unresolved_alerts || []).length})`}>
            {(bd.unresolved_alerts || []).length === 0
              ? <div style={{ color: '#4caf50', fontWeight: 600 }}>✓ No unresolved alerts</div>
              : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#fff5f5' }}>
                      {['Alert Type', 'Severity', 'Device', 'Gateway', 'Patient', 'Acknowledged', 'Timestamp'].map(h => (
                        <th key={h} style={{ padding: '6px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.unresolved_alerts || []).map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#fafafa' : '#fff' }}>
                        <td style={{ padding: '5px 10px' }}>{a.alert_type}</td>
                        <td style={{ padding: '5px 10px' }}><SevBadge sev={a.severity} /></td>
                        <td style={{ padding: '5px 10px', fontFamily: 'monospace' }}>{a.device_id ?? '–'}</td>
                        <td style={{ padding: '5px 10px', fontFamily: 'monospace' }}>{a.gateway_id ?? '–'}</td>
                        <td style={{ padding: '5px 10px' }}>{a.patient_id ?? '–'}</td>
                        <td style={{ padding: '5px 10px' }}>
                          {a.acknowledged
                            ? <span style={{ color: '#4caf50' }}>Yes</span>
                            : <span style={{ color: '#f44336' }}>No</span>}
                        </td>
                        <td style={{ padding: '5px 10px', color: '#64748b' }}>{a.timestamp ? a.timestamp.replace('T', ' ').slice(0, 16) : '–'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
          </Card>
        </>
      )}

      {/* ── PATIENTS TAB ── */}
      {tab === 'patients' && (
        <Card title={`Patient Device Coverage (${(bd.patient_coverage || []).length} patients)`}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient ID', 'Devices', 'Online', 'Device Types', 'Avg Battery %'].map(h => (
                  <th key={h} style={{ padding: '6px 12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(bd.patient_coverage || []).map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#fafafa' : '#fff' }}>
                  <td style={{ padding: '6px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{p.devices}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', color: p.online === p.devices ? '#4caf50' : '#ff9800' }}>{p.online}</td>
                  <td style={{ padding: '6px 12px' }}>
                    {(p.types || []).map((t, j) => (
                      <span key={j} style={{ background: '#e0f2fe', color: '#0369a1', borderRadius: 10, padding: '1px 7px', fontSize: 10, margin: '1px 2px', display: 'inline-block' }}>
                        {t}
                      </span>
                    ))}
                  </td>
                  <td style={{ padding: '6px 12px', color: p.avg_battery < 20 ? '#f44336' : '#1e293b', fontWeight: p.avg_battery < 20 ? 700 : 400 }}>{num(p.avg_battery)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Device Types">
              {Object.entries(df.device_types || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 10 }}>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#1e88e5' }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{v}</div>
                </div>
              ))}
            </Card>
            <Card title="Connectivity Modes">
              {Object.entries(df.connectivity_modes || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 10 }}>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#7c4dff', textTransform: 'capitalize' }}>{k}</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{v}</div>
                </div>
              ))}
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Alert Types">
              {Object.entries(df.alert_types || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 10 }}>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#f44336' }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{v}</div>
                </div>
              ))}
            </Card>
            <Card title="Severity Levels">
              {Object.entries(df.severity_levels || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 10 }}>
                  <div style={{ fontWeight: 600, fontSize: 12, color: SEV_COLORS[k] || '#333', textTransform: 'capitalize' }}>{k}</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{v}</div>
                </div>
              ))}
            </Card>
          </div>

          <Card title="KPI Definitions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '6px 12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0' }}>KPI</th>
                  <th style={{ padding: '6px 12px', textAlign: 'left', fontWeight: 600, borderBottom: '2px solid #e2e8f0' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(df.kpi_definitions || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#fafafa' : '#fff' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600 }}>{r.kpi}</td>
                    <td style={{ padding: '6px 12px', color: '#475569' }}>{r.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical & Regulatory References">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(df.clinical_references || []).map((ref, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{ref}</li>
              ))}
            </ul>
          </Card>
        </>
      )}
    </div>
  )
}

// safe float helper (client-side only, not imported)
function _safe_float(v, d = 0) {
  if (v == null) return d
  const f = parseFloat(v)
  return isNaN(f) ? d : f
}
