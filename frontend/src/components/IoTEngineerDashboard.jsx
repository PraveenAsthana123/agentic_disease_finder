import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || 'http://localhost:8010'

const STATUS_COLORS = {
  online: '#22c55e',
  offline: '#ef4444',
  maintenance: '#f59e0b',
}

const ALERT_SEVERITY_COLORS = {
  critical: '#ef4444',
  warning: '#f59e0b',
  info: '#3b82f6',
}

const PIE_COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Card({ title, children, span, accent }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined,
      borderLeft: accent ? `4px solid ${accent}` : undefined,
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

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color,
      textTransform: 'capitalize'
    }}>{status || 'Unknown'}</span>
  )
}

function SeverityBadge({ severity }) {
  const color = ALERT_SEVERITY_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color,
      textTransform: 'capitalize'
    }}>{severity || 'Unknown'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'device_fleet', label: 'Device Fleet' },
  { id: 'gateway_health', label: 'Gateway Health' },
  { id: 'patients', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function IoTEngineerDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [expandedDevice, setExpandedDevice] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/iot-engineer/overview`),
      axios.get(`${API_URL}/api/iot-engineer/breakdown`),
      axios.get(`${API_URL}/api/iot-engineer/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefs(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading IoT Engineer data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const kpis = ov.kpis || {}
  const deviceStatusDist = ov.device_status_distribution || []
  const batteryHistogram = ov.battery_histogram || []
  const alertTypeDist = ov.alert_type_distribution || []
  const gatewayUptimeData = ov.gateway_uptime_data || []

  const firmwareDist = bd.firmware_distribution || []
  const deviceTypeDist = bd.device_type_distribution || []
  const deviceBatteryHist = bd.device_battery_histogram || []
  const signalStrengthDist = bd.signal_strength_distribution || []
  const gateways = bd.gateways || []
  const devices = bd.devices || []

  const thStyle = {
    textAlign: 'left', padding: '8px 10px', fontSize: 12,
    color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600
  }
  const tdStyle = {
    padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9'
  }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        IoT Engineer Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Biomedical device fleet management: device status, gateway health, stream latency, battery monitoring, and alert tracking
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13,
            fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b',
            background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── Overview Tab ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>

          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
              <KPI label="Devices Online" value={fmt(kpis.devices_online)}
                color="#22c55e" sub="Currently connected" />
              <KPI label="Total Devices" value={fmt(kpis.devices_total)}
                sub="Registered in fleet" />
              <KPI label="Online Rate" value={`${fmt(kpis.online_pct)}%`}
                color={kpis.online_pct >= 90 ? '#22c55e' : kpis.online_pct >= 70 ? '#f59e0b' : '#ef4444'}
                sub="Target >= 90%" />
              <KPI label="Gateway Count" value={fmt(kpis.gateway_count)}
                sub="Active gateways" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Avg Gateway Uptime" value={`${fmt(kpis.avg_gateway_uptime_pct)}%`}
                color={kpis.avg_gateway_uptime_pct >= 95 ? '#22c55e' : kpis.avg_gateway_uptime_pct >= 80 ? '#f59e0b' : '#ef4444'}
                sub="Target >= 95%" />
              <KPI label="Avg Stream Latency" value={`${fmt(kpis.avg_stream_latency_ms)} ms`}
                color={kpis.avg_stream_latency_ms <= 50 ? '#22c55e' : kpis.avg_stream_latency_ms <= 200 ? '#f59e0b' : '#ef4444'}
                sub="Target <= 50 ms" />
              <KPI label="SOS Alerts" value={fmt(kpis.sos_alerts_total)}
                color={kpis.sos_alerts_total > 0 ? '#ef4444' : '#22c55e'}
                sub="Total SOS events" />
              <KPI label="Low Battery / Signal" value={fmt(kpis.battery_signal_low_count)}
                color={kpis.battery_signal_low_count > 5 ? '#ef4444' : kpis.battery_signal_low_count > 0 ? '#f59e0b' : '#22c55e'}
                sub="Devices needing attention" />
            </div>
          </Card>

          {/* Device Status Pie Chart */}
          <Card title="Device Status Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={deviceStatusDist.filter(d => d.count > 0)} dataKey="count" nameKey="status"
                  cx="50%" cy="50%" outerRadius={80}
                  label={({ status, count }) => `${status}: ${count}`}>
                  {deviceStatusDist.map((d, i) => (
                    <Cell key={i} fill={STATUS_COLORS[d.status] || PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Battery Level Histogram */}
          <Card title="Battery Level Distribution (%)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={batteryHistogram}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {batteryHistogram.map((d, i) => {
                    const lo = parseInt(d.range)
                    const color = lo >= 60 ? '#22c55e' : lo >= 30 ? '#f59e0b' : '#ef4444'
                    return <Cell key={i} fill={color} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Green &gt;= 60% | Yellow 30-59% | Red &lt; 30%
            </div>
          </Card>

          {/* Alert Type Distribution */}
          <Card title="Alert Type Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={alertTypeDist.filter(d => d.count > 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {alertTypeDist.map((d, i) => (
                    <Cell key={i} fill={ALERT_SEVERITY_COLORS[d.severity] || PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Gateway Uptime Bar Chart */}
          <Card title="Gateway Uptime (%)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={gatewayUptimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gateway_id" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="uptime_pct" name="Uptime" radius={[4, 4, 0, 0]}>
                  {gatewayUptimeData.map((d, i) => (
                    <Cell key={i}
                      fill={d.uptime_pct >= 95 ? '#22c55e' : d.uptime_pct >= 80 ? '#f59e0b' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Green &gt;= 95% | Yellow 80-94% | Red &lt; 80%
            </div>
          </Card>
        </div>
      )}

      {/* ─── Device Fleet Tab ─── */}
      {tab === 'device_fleet' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>

          {/* Firmware Version Distribution */}
          <Card title="Firmware Version Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={firmwareDist.filter(d => d.count > 0)} dataKey="count" nameKey="version"
                  cx="50%" cy="50%" outerRadius={90}
                  label={({ version, count }) => `${version}: ${count}`}>
                  {firmwareDist.map((d, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Device Type Distribution */}
          <Card title="Device Type Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={deviceTypeDist.filter(d => d.count > 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {deviceTypeDist.map((d, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Device Battery Histogram */}
          <Card title="Per-Device Battery Level Histogram" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={deviceBatteryHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }}
                  label={{ value: 'Battery %', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                  {deviceBatteryHist.map((d, i) => {
                    const lo = parseInt(d.range)
                    const color = lo >= 60 ? '#22c55e' : lo >= 30 ? '#f59e0b' : '#ef4444'
                    return <Cell key={i} fill={color} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Signal Strength Distribution */}
          <Card title="Signal Strength Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={signalStrengthDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {signalStrengthDist.map((d, i) => {
                    const lo = parseInt(d.range)
                    const color = lo >= 70 ? '#22c55e' : lo >= 40 ? '#f59e0b' : '#ef4444'
                    return <Cell key={i} fill={color} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Green &gt;= 70% | Yellow 40-69% | Red &lt; 40%
            </div>
          </Card>
        </div>
      )}

      {/* ─── Gateway Health Tab ─── */}
      {tab === 'gateway_health' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {gateways.length === 0 ? (
            <Card span={2}>
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No gateway data available from the API.</p>
            </Card>
          ) : (
            gateways.map((gw, i) => {
              const uptimeColor = gw.uptime_pct >= 95 ? '#22c55e' : gw.uptime_pct >= 80 ? '#f59e0b' : '#ef4444'
              return (
                <Card key={i} accent={uptimeColor}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <div>
                      <span style={{ fontWeight: 700, fontSize: 15, color: '#0f172a' }}>{gw.gateway_id || `Gateway ${i + 1}`}</span>
                      <span style={{ marginLeft: 10 }}><StatusBadge status={gw.status} /></span>
                    </div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: uptimeColor }}>
                      {fmt(gw.uptime_pct)}%
                    </div>
                  </div>
                  <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8,
                    fontSize: 12, color: '#475569'
                  }}>
                    <div><span style={{ color: '#94a3b8' }}>Connected Devices:</span> <strong>{fmt(gw.connected_devices)}</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>Location:</span> <strong>{gw.location || '--'}</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>Firmware:</span> <strong>{gw.firmware_version || '--'}</strong></div>
                    <div><span style={{ color: '#94a3b8' }}>Last Seen:</span> <strong>{gw.last_seen || '--'}</strong></div>
                  </div>
                </Card>
              )
            })
          )}
        </div>
      )}

      {/* ─── Patient Detail Tab ─── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gap: 12 }}>
          {devices.length === 0 ? (
            <Card>
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No device data available from the API.</p>
            </Card>
          ) : (
            devices.map((dev, i) => {
              const isExpanded = expandedDevice === dev.device_id
              return (
                <Card key={i}>
                  <div
                    style={{ display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer' }}
                    onClick={() => setExpandedDevice(isExpanded ? null : dev.device_id)}
                  >
                    <span style={{
                      fontSize: 18,
                      transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                      transition: 'transform 0.2s',
                      display: 'inline-block'
                    }}>&#9654;</span>
                    <div style={{ flex: 1 }}>
                      <span style={{ fontWeight: 600, fontSize: 14 }}>{dev.device_id || `Device ${i + 1}`}</span>
                      <span style={{ color: '#94a3b8', fontSize: 12, marginLeft: 10 }}>
                        {dev.device_type || '--'} | Patient: {dev.patient_id || '--'}
                      </span>
                    </div>
                    <StatusBadge status={dev.status} />
                    <span style={{ fontSize: 12, color: '#64748b' }}>
                      Battery: <strong style={{
                        color: dev.battery_pct >= 60 ? '#22c55e' : dev.battery_pct >= 30 ? '#f59e0b' : '#ef4444'
                      }}>{fmt(dev.battery_pct)}%</strong>
                    </span>
                    <span style={{ fontSize: 12, color: '#64748b' }}>
                      Signal: <strong style={{
                        color: dev.signal_strength >= 70 ? '#22c55e' : dev.signal_strength >= 40 ? '#f59e0b' : '#ef4444'
                      }}>{fmt(dev.signal_strength)}%</strong>
                    </span>
                  </div>

                  {isExpanded && (
                    <div style={{ marginTop: 16 }}>
                      {/* Device details */}
                      <div style={{
                        display: 'flex', gap: 20, flexWrap: 'wrap',
                        marginBottom: 16, padding: '10px 14px',
                        background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#475569'
                      }}>
                        <span><strong>Device ID:</strong> {dev.device_id || '--'}</span>
                        <span><strong>Patient ID:</strong> {dev.patient_id || '--'}</span>
                        <span><strong>Device Type:</strong> {dev.device_type || '--'}</span>
                        <span><strong>Battery:</strong> {fmt(dev.battery_pct)}%</span>
                        <span><strong>Signal Strength:</strong> {fmt(dev.signal_strength)}%</span>
                        <span><strong>Status:</strong> <StatusBadge status={dev.status} /></span>
                        <span><strong>Assigned Gateway:</strong> {dev.assigned_gateway || '--'}</span>
                        <span><strong>Firmware:</strong> {dev.firmware_version || '--'}</span>
                        <span><strong>Last Seen:</strong> {dev.last_seen || '--'}</span>
                      </div>

                      {/* Device alerts */}
                      <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>
                        Alerts ({(dev.alerts || []).length} events)
                      </h4>
                      {(dev.alerts || []).length > 0 ? (
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                          <thead>
                            <tr>
                              <th style={thStyle}>Alert Type</th>
                              <th style={thStyle}>Severity</th>
                              <th style={thStyle}>Message</th>
                              <th style={thStyle}>Timestamp</th>
                              <th style={thStyle}>Resolved</th>
                            </tr>
                          </thead>
                          <tbody>
                            {dev.alerts.map((alert, j) => (
                              <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>{alert.type || '--'}</td>
                                <td style={tdStyle}><SeverityBadge severity={alert.severity} /></td>
                                <td style={{ ...tdStyle, fontSize: 12 }}>{alert.message || '--'}</td>
                                <td style={{ ...tdStyle, fontSize: 12 }}>{alert.timestamp || '--'}</td>
                                <td style={tdStyle}>
                                  <Badge
                                    text={alert.resolved ? 'Resolved' : 'Open'}
                                    color={alert.resolved ? '#22c55e' : '#ef4444'}
                                  />
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <p style={{ fontSize: 12, color: '#94a3b8' }}>No alerts for this device.</p>
                      )}
                    </div>
                  )}
                </Card>
              )
            })
          )}
        </div>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gap: 16 }}>
          {defs?.title && (
            <div style={{ padding: '12px 16px', background: '#eff6ff', borderRadius: 8, borderLeft: '4px solid #2563eb' }}>
              <h3 style={{ margin: 0, fontSize: 15, color: '#1e40af' }}>{defs.title}</h3>
            </div>
          )}
          {(defs?.sections || []).map((section, si) => (
            <Card key={si} title={section.heading}>
              {Array.isArray(section.items) && section.items.length > 0 ? (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ ...thStyle, width: '28%' }}>Term</th>
                      <th style={thStyle}>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.items.map((item, ii) => (
                      <tr key={ii} style={{ background: ii % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>{item.term}</td>
                        <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.6 }}>{item.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : section.description ? (
                <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{section.description}</p>
              ) : null}
            </Card>
          ))}
          {(!defs?.sections || defs.sections.length === 0) && !defs?.concepts && (
            <Card title="IoT & Biomedical Device Monitoring">
              {(defs?.concepts || []).length > 0 ? (
                (defs.concepts || []).map((item, i) => (
                  <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                    <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                  </div>
                ))
              ) : (
                <p style={{ fontSize: 13, color: '#94a3b8' }}>No definition data available from the API.</p>
              )}
            </Card>
          )}
          {defs?.concepts && (!defs?.sections || defs.sections.length === 0) && (
            <Card title="IoT & Biomedical Device Monitoring">
              {(defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))}
            </Card>
          )}
        </div>
      )}

      <div style={{
        marginTop: 24, padding: 16, background: '#f1f5f9',
        borderRadius: 8, fontSize: 12, color: '#64748b'
      }}>
        IoT Engineer Dashboard — Biomedical device fleet monitoring ({fmt(kpis.devices_total)} devices,
        {' '}{fmt(kpis.gateway_count)} gateways) |
        Real-time device status, battery, signal strength, and alert tracking
      </div>
    </div>
  )
}
