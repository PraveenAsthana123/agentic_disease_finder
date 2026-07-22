import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = {
  built: '#4caf50', live: '#4caf50', partial: '#ff9800',
  'needs-credentials': '#1e88e5', planned: '#f44336', missing: '#f44336',
  'n/a': '#94a3b8'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Badge({ status }) {
  const bg = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

function Card({ title, children, style }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, border: '1px solid #e2e8f0',
      padding: 16, ...style
    }}>
      {title && <div style={{ fontWeight: 600, fontSize: 13, color: '#475569', marginBottom: 8 }}>{title}</div>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <Card>
      <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{fmt(value)}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </Card>
  )
}

export default function IntegrationRoleDashboard() {
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
          axios.get(`${API_URL}/api/integration/overview`),
          axios.get(`${API_URL}/api/integration/breakdown`),
          axios.get(`${API_URL}/api/integration/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Integration data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      Loading Integration Dashboard...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>
      Error: {error}
    </div>
  )

  const tabs = ['overview', 'integrations', 'devices', 'gaps', 'definitions']
  const s = overview?.summary || {}

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ margin: 0, fontSize: 20, color: '#1e293b' }}>
        Integration Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '4px 0 16px' }}>
        EMR/FHIR, device streaming, API contracts, webhooks — integration readiness across external systems and IoT devices
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 16px', border: 'none', cursor: 'pointer',
              background: tab === t ? '#1e293b' : 'transparent',
              color: tab === t ? '#fff' : '#64748b',
              borderRadius: '6px 6px 0 0', fontWeight: tab === t ? 600 : 400,
              fontSize: 13, textTransform: 'capitalize'
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
            <KPI label="Total Integrations" value={s.total_integrations} />
            <KPI label="Total Devices" value={s.total_devices} />
            <KPI label="Integrations Live" value={s.integrations_live} />
            <KPI label="Needs Credentials" value={s.integrations_needs_credentials} />
            <KPI label="Delivery Channels" value={s.delivery_channels_total} sub={`${s.delivery_channels_live} live`} />
            <KPI label="Devices Partial" value={s.devices_partial} />
            <KPI label="Devices Planned" value={s.devices_planned} />
            <KPI label="API Contracts" value={s.api_contracts_registered} />
            <KPI label="Overall Readiness" value={`${s.overall_readiness_pct}%`} />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Integration Status">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview?.integration_status_distribution || []} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {(overview?.integration_status_distribution || []).map((e, i) => (
                      <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Delivery Channel Status">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview?.delivery_channel_distribution || []} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {(overview?.delivery_channel_distribution || []).map((e, i) => (
                      <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Device Status">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview?.device_status_distribution || []} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {(overview?.device_status_distribution || []).map((e, i) => (
                      <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Integrations by Category">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview?.integration_by_category || []} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="category" type="category" width={100} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </div>
      )}

      {/* ── Integrations Tab ── */}
      {tab === 'integrations' && (
        <div>
          <Card title="External Integrations" style={{ marginBottom: 16 }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Name', 'Category', 'Purpose', 'Status', 'Config'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.integrations || []).map((item, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{item.name}</td>
                      <td style={{ padding: '8px 12px' }}>{item.category}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{item.purpose}</td>
                      <td style={{ padding: '8px 12px' }}><Badge status={item.status} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 10, color: '#94a3b8', fontFamily: 'monospace' }}>{item.config}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Delivery Channels">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Name', 'Purpose', 'Status', 'Config', 'Note'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.delivery_channels || []).map((item, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{item.name}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{item.purpose}</td>
                      <td style={{ padding: '8px 12px' }}><Badge status={item.status} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 10, color: '#94a3b8', fontFamily: 'monospace' }}>{item.config}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{item.note || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Devices Tab ── */}
      {tab === 'devices' && (
        <div>
          <Card title="IoT Device Fleet">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Name', 'Type', 'Channels', 'Modes', 'Data', 'Status', 'Alert'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.devices || []).map((dev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{dev.name}</td>
                      <td style={{ padding: '8px 12px' }}>{dev.type}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{dev.channels ?? '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11 }}>{(dev.modes || []).join(', ')}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{(dev.data || []).join(', ')}</td>
                      <td style={{ padding: '8px 12px' }}><Badge status={dev.status} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{dev.alert || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Gaps Tab ── */}
      {tab === 'gaps' && (
        <div>
          <Card title="Stakeholder Integration Gaps" style={{ marginBottom: 16 }}>
            {(breakdown?.stakeholder_integration_gaps || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No integration gaps found.</p>
            ) : (
              (breakdown?.stakeholder_integration_gaps || []).map((sh, i) => (
                <div key={i} style={{ marginBottom: 16, padding: 12, background: '#f8fafc', borderRadius: 6 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6 }}>{sh.role} ({sh.gap_count} gaps)</div>
                  <ul style={{ margin: 0, paddingLeft: 20 }}>
                    {sh.integration_gaps.map((gap, j) => (
                      <li key={j} style={{ fontSize: 12, color: '#475569', marginBottom: 2 }}>{gap}</li>
                    ))}
                  </ul>
                </div>
              ))
            )}
          </Card>

          <Card title="Admin Module Integrations (MCP Connectors)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Label', 'Via', 'Purpose', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.admin_integrations || []).map((item, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{item.label}</td>
                      <td style={{ padding: '8px 12px' }}>{item.via || '--'}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{item.purpose}</td>
                      <td style={{ padding: '8px 12px' }}><Badge status={item.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && (
        <div>
          {(defs?.definitions || []).map((d, i) => (
            <Card key={i} style={{ marginBottom: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{d.term}</div>
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{d.definition}</div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
