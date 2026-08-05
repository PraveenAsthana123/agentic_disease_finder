import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
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

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4']

const STATUS_COLOR = { active: '#10b981', offline: '#ef4444', charging: '#f59e0b', maintenance: '#8b5cf6' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Device Details' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EmotivWearableDashboard() {
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
      axios.get(`${API_URL}/api/emotiv-wearable/overview`),
      axios.get(`${API_URL}/api/emotiv-wearable/breakdown`),
      axios.get(`${API_URL}/api/emotiv-wearable/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Emotiv Wearable data…</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Emotiv Wearable Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          EEG-class wearable devices — Empatica Embrace2 · Byteflies Sensor Dot · BioStampRC
          · seizure detection · electrode contact quality · battery health
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16 }}>

          {/* KPI row */}
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)', gap: 12 }}>
              <KPI label="Total Devices" value={k.total_devices} />
              <KPI label="Active" value={k.active_devices} color="#10b981" />
              <KPI label="Offline" value={k.offline_devices} color="#ef4444" />
              <KPI label="Low Battery" value={k.low_battery_devices} color="#f59e0b"
                sub="< 30%" />
              <KPI label="Avg Battery" value={`${k.avg_battery_pct}%`} />
              <KPI label="Outdated FW" value={k.outdated_firmware_count} color="#8b5cf6" />
            </div>
          </Card>

          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12 }}>
              <KPI label="Sessions" value={k.total_sessions} />
              <KPI label="Seizures Detected" value={k.seizures_detected} color="#ef4444" />
              <KPI label="Detection Rate" value={`${k.seizure_detection_rate_pct}%`} />
              <KPI label="Avg Confidence" value={k.avg_detection_confidence} />
              <KPI label="Avg Health Score" value={k.avg_health_score} color="#3b82f6" />
            </div>
          </Card>

          {/* Brand distribution */}
          <Card title="Device Brand Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview?.brand_distribution || []} dataKey="count" nameKey="brand"
                  cx="50%" cy="50%" outerRadius={75} label={({ brand, count }) => `${brand} (${count})`}>
                  {(overview?.brand_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Connectivity */}
          <Card title="Connectivity Modes" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview?.connectivity_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mode" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Electrode channel quality */}
          <Card title="Electrode Contact Quality — Emotiv EPOC+ 14-Channel" span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7,1fr)', gap: 8 }}>
              {(overview?.channel_quality || []).map(ch => (
                <div key={ch.channel} style={{
                  background: ch.status === 'good' ? '#dcfce7' : '#fef9c3',
                  border: `1px solid ${ch.status === 'good' ? '#86efac' : '#fde68a'}`,
                  borderRadius: 8, padding: '10px 6px', textAlign: 'center'
                }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b' }}>{ch.channel}</div>
                  <div style={{ fontSize: 18, fontWeight: 700,
                    color: ch.status === 'good' ? '#16a34a' : '#d97706' }}>{ch.quality_pct}%</div>
                  <div style={{ fontSize: 10, color: '#64748b' }}>{ch.status}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Firmware */}
          <Card title="Firmware Version Distribution" span={4}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {(overview?.firmware_distribution || []).map((f, i) => (
                <div key={f.version} style={{
                  background: i === 0 ? '#eff6ff' : '#fef2f2',
                  border: `1px solid ${i === 0 ? '#bfdbfe' : '#fecaca'}`,
                  borderRadius: 8, padding: '6px 14px', fontSize: 12
                }}>
                  <span style={{ fontWeight: 700 }}>{f.version}</span>
                  <span style={{ marginLeft: 8, color: '#64748b' }}>{f.count} devices</span>
                  {i === 0 && <span style={{ marginLeft: 6, color: '#2563eb', fontSize: 10 }}>latest</span>}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ── BREAKDOWN TAB ── */}
      {tab === 'breakdown' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 16 }}>

          {/* Device table */}
          <Card title="Device Inventory" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'Device', 'Brand', 'Status', 'Battery', 'Connectivity',
                      'Firmware', 'Sessions', 'Seizures', 'Confidence'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left',
                        fontWeight: 600, color: '#64748b', borderBottom: '1px solid #e2e8f0' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.device_table || []).map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '7px 10px', fontWeight: 600 }}>{d.patient_id}</td>
                      <td style={{ padding: '7px 10px', color: '#475569' }}>{d.device_id}</td>
                      <td style={{ padding: '7px 10px' }}>{d.brand}</td>
                      <td style={{ padding: '7px 10px' }}>
                        <span style={{
                          background: `${STATUS_COLOR[d.status] || '#94a3b8'}22`,
                          color: STATUS_COLOR[d.status] || '#64748b',
                          borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 600
                        }}>{d.status}</span>
                      </td>
                      <td style={{ padding: '7px 10px' }}>
                        <span style={{ color: d.battery_level < 30 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                          {d.battery_level}%
                        </span>
                      </td>
                      <td style={{ padding: '7px 10px', color: '#475569' }}>{d.connectivity}</td>
                      <td style={{ padding: '7px 10px', color: '#475569' }}>{d.firmware_version}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'right' }}>{d.total_sessions}</td>
                      <td style={{ padding: '7px 10px', textAlign: 'right',
                        color: d.seizures_detected > 0 ? '#ef4444' : '#64748b', fontWeight: 600 }}>
                        {d.seizures_detected}
                      </td>
                      <td style={{ padding: '7px 10px', textAlign: 'right' }}>{d.avg_confidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Battery distribution */}
          <Card title="Battery Level Buckets">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown?.battery_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {(breakdown?.battery_distribution || []).map((d, i) => (
                    <Cell key={i} fill={
                      d.bucket === '<30%' ? '#ef4444' :
                      d.bucket === '30-60%' ? '#f59e0b' :
                      d.bucket === '60-90%' ? '#3b82f6' : '#10b981'
                    } />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Patient seizure confidence */}
          <Card title="Per-Patient Avg Seizure Confidence">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={(breakdown?.patient_confidence || []).slice(0, 15)}
                layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} />
                <YAxis dataKey="patient_id" type="category" width={70} tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v) => v.toFixed(3)} />
                <Bar dataKey="avg_confidence" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 16 }}>

          <Card title="About This Dashboard" span={2}>
            <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
              {definitions?.overview}
            </p>
          </Card>

          {/* Device model table */}
          <Card title="Supported Device Models" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Brand / Model', 'Type', 'Channels', 'Sample Rate', 'Connectivity',
                    'FDA Cleared', 'Detection', 'Battery'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left',
                      fontWeight: 600, color: '#64748b', borderBottom: '1px solid #e2e8f0' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(definitions?.device_models || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 600 }}>{m.brand}</td>
                    <td style={{ padding: '7px 10px', color: '#475569' }}>{m.type}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{m.channels}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{m.sample_rate_hz} Hz</td>
                    <td style={{ padding: '7px 10px' }}>{m.connectivity}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>
                      <span style={{ color: m.fda_cleared ? '#16a34a' : '#64748b' }}>
                        {m.fda_cleared ? '✓ Yes' : '—'}
                      </span>
                    </td>
                    <td style={{ padding: '7px 10px', color: '#475569' }}>{m.detection}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{m.battery_hours}h</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Metric glossary */}
          <Card title="Metric Definitions">
            <dl style={{ margin: 0 }}>
              {Object.entries(definitions?.metrics || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 10 }}>
                  <dt style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{k.replace(/_/g, ' ')}</dt>
                  <dd style={{ margin: '2px 0 0 0', fontSize: 12, color: '#475569' }}>{v}</dd>
                </div>
              ))}
            </dl>
          </Card>

          {/* EPOC+ channels */}
          <Card title="Emotiv EPOC+ 14-Channel Layout">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {(definitions?.epoc_channels || []).map((ch, i) => (
                <span key={ch} style={{
                  background: '#eff6ff', border: '1px solid #bfdbfe',
                  borderRadius: 6, padding: '4px 10px', fontSize: 12, fontWeight: 600, color: '#1d4ed8'
                }}>{i + 1}. {ch}</span>
              ))}
            </div>
            <p style={{ margin: '12px 0 0', fontSize: 11, color: '#64748b' }}>
              Based on the 10-20 international EEG electrode system. Positions shown above correspond
              to Emotiv EPOC+ research headset. Actual clinical devices (Embrace2, Byteflies) use
              fewer channels but the same spatial reference.
            </p>
          </Card>

          {/* References */}
          <Card title="Clinical References" span={2}>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(definitions?.clinical_references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{r}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
