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
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 30, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 3 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{text}</span>
  )
}

const MODE_COLOR = { online: '#10b981', offline: '#64748b', batch: '#f59e0b' }
const QUALITY_COLOR = { good: '#10b981', fair: '#f59e0b', poor: '#ef4444' }
const JOB_COLOR = { queued: '#3b82f6', processing: '#f59e0b', done: '#10b981', failed: '#ef4444' }
const COLORS = ['#10b981', '#64748b', '#f59e0b', '#3b82f6', '#8b5cf6']

const TABS = ['Overview', 'Devices', 'Batch Queue', 'Definitions']

export default function DeviceModeDashboard() {
  const [tab, setTab] = useState('Overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/device-mode/overview`),
      axios.get(`${API_URL}/api/device-mode/breakdown`),
      axios.get(`${API_URL}/api/device-mode/definitions`),
    ]).then(([ov, br, df]) => {
      setOverview(ov.data)
      setBreakdown(br.data)
      setDefinitions(df.data)
      setErr(null)
    }).catch(e => setErr(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading…</div>
  if (err) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {err}</div>

  const kpis = overview?.kpis || {}
  const devices = breakdown?.devices || []
  const batchQueue = breakdown?.batch_queue || []

  return (
    <div style={{ padding: 24, fontFamily: 'Inter, system-ui, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <span style={{ fontSize: 26 }}>📡</span>
        <div>
          <h2 style={{ margin: 0, fontSize: 20, color: '#1e293b' }}>Device Mode Manager</h2>
          <div style={{ fontSize: 12, color: '#64748b' }}>Online streaming · Offline sync · Batch processing</div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t ? '#3b82f6' : 'transparent',
            color: tab === t ? '#fff' : '#64748b',
            borderRadius: '6px 6px 0 0',
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview' && (
        <div>
          {/* KPI row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 14, marginBottom: 20 }}>
            {[
              { label: 'Total Devices', value: kpis.total_devices, color: '#1e293b' },
              { label: 'Online', value: kpis.online, color: '#10b981', sub: 'streaming live' },
              { label: 'Offline', value: kpis.offline, color: '#64748b', sub: 'pending sync' },
              { label: 'Batch', value: kpis.batch, color: '#f59e0b', sub: 'long-term rec' },
              { label: 'Avg Battery', value: kpis.avg_battery_pct != null ? `${kpis.avg_battery_pct}%` : '--', color: '#8b5cf6' },
              { label: 'Avg Stream Hz', value: kpis.avg_stream_hz ? `${kpis.avg_stream_hz} Hz` : '--', color: '#3b82f6', sub: 'online only' },
            ].map((k, i) => (
              <Card key={i}><KPI {...k} /></Card>
            ))}
          </div>

          {/* Charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 }}>
            <Card title="Devices by Mode">
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={overview.by_mode} dataKey="count" nameKey="mode" cx="50%" cy="50%" outerRadius={65} label={({ mode, count }) => `${mode}: ${count}`}>
                    {overview.by_mode.map((e, i) => (
                      <Cell key={i} fill={MODE_COLOR[e.mode] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Signal Quality">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={overview.by_quality} margin={{ left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="quality" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4,4,0,0]}>
                    {overview.by_quality.map((e, i) => (
                      <Cell key={i} fill={QUALITY_COLOR[e.quality] || '#3b82f6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Batch Queue">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
                {Object.entries(overview.batch_queue_summary || {}).map(([status, count]) => (
                  <div key={status} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Badge text={status} color={JOB_COLOR[status] || '#64748b'} />
                    <span style={{ fontWeight: 700, fontSize: 18, color: JOB_COLOR[status] || '#1e293b' }}>{count}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          <Card title="Device Type Distribution">
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={overview.by_type} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="type" type="category" tick={{ fontSize: 11 }} width={130} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0,4,4,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'Devices' && (
        <Card title={`All Devices (${devices.length})`} span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Device ID', 'Label', 'Type', 'Mode', 'Stream Hz', 'Battery', 'Signal', 'Patient', 'Last Sync'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {devices.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>{d.device_id}</td>
                    <td style={{ padding: '8px 12px' }}>{d.label}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{d.type}</td>
                    <td style={{ padding: '8px 12px' }}><Badge text={d.mode} color={MODE_COLOR[d.mode] || '#64748b'} /></td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>{d.stream_hz ? `${d.stream_hz} Hz` : '—'}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                      <span style={{ color: d.battery_pct < 40 ? '#ef4444' : d.battery_pct < 70 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                        {d.battery_pct}%
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}><Badge text={d.signal_quality} color={QUALITY_COLOR[d.signal_quality] || '#64748b'} /></td>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 11 }}>{d.patient}</td>
                    <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{d.last_sync?.replace('T', ' ').replace('Z', ' UTC')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'Batch Queue' && (
        <Card title={`Batch Processing Queue (${batchQueue.length} jobs)`} span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Job ID', 'Device', 'Patient', 'Duration', 'Size (MB)', 'Status', 'ETA'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {batchQueue.map((j, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>{j.job_id}</td>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>{j.device_id}</td>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>{j.patient}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>{j.duration_h}h</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>{j.size_mb} MB</td>
                    <td style={{ padding: '8px 12px' }}><Badge text={j.status} color={JOB_COLOR[j.status] || '#64748b'} /></td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>{j.eta_min > 0 ? `${j.eta_min} min` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'Definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Device Modes" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {definitions.modes.map((m, i) => (
                <div key={i} style={{ padding: 14, borderRadius: 8, background: (MODE_COLOR[m.name] || '#64748b') + '12', border: `1px solid ${(MODE_COLOR[m.name] || '#64748b')}30` }}>
                  <div style={{ fontWeight: 700, color: MODE_COLOR[m.name] || '#64748b', marginBottom: 6 }}>{m.name.toUpperCase()}</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{m.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Signal Quality Grades">
            {Object.entries(definitions.signal_quality || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 10, marginBottom: 10, alignItems: 'flex-start' }}>
                <Badge text={k} color={QUALITY_COLOR[k] || '#64748b'} />
                <span style={{ fontSize: 12, color: '#475569' }}>{v}</span>
              </div>
            ))}
          </Card>

          <Card title="Batch Job Statuses">
            {definitions.batch_statuses?.map(s => (
              <div key={s} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'center' }}>
                <Badge text={s} color={JOB_COLOR[s] || '#64748b'} />
              </div>
            ))}
          </Card>

          <Card title="Sync Policy" span={2}>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{definitions.sync_policy}</div>
          </Card>
        </div>
      )}
    </div>
  )
}
