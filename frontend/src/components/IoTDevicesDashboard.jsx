import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b', '#795548', '#9e9e9e']
const STATUS_COLORS = { Built: '#4caf50', Partial: '#ff9800', Planned: '#f44336', Unknown: '#9e9e9e' }

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const s = (status || 'unknown').charAt(0).toUpperCase() + (status || 'unknown').slice(1)
  const bg = STATUS_COLORS[s] || '#9e9e9e'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {s}
    </span>
  )
}

function ModeBadge({ mode }) {
  const modeColors = { online: '#4caf50', offline: '#ff9800', hybrid: '#7c4dff' }
  const bg = modeColors[mode] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '1px 6px', fontSize: 10, fontWeight: 600, margin: '1px 2px', display: 'inline-block'
    }}>
      {mode}
    </span>
  )
}

function Pill({ text }) {
  return (
    <span style={{
      background: '#f1f5f9', color: '#475569', borderRadius: 12, padding: '2px 8px',
      fontSize: 11, display: 'inline-block', margin: '2px 3px'
    }}>
      {text}
    </span>
  )
}

function BoolBadge({ val }) {
  return val
    ? <span style={{ color: '#4caf50', fontWeight: 600, fontSize: 13 }}>Yes</span>
    : <span style={{ color: '#e0e0e0', fontSize: 13 }}>-</span>
}

export default function IoTDevicesDashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const tabs = ['overview', 'by-type', 'device-matrix', 'connectivity', 'definitions']

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/iot-devices/overview`),
      axios.get(`${API_URL}/iot-devices/breakdown`),
      axios.get(`${API_URL}/iot-devices/definitions`),
    ])
      .then(([ov, bd, df]) => setData({ overview: ov.data, breakdown: bd.data, definitions: df.data }))
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ color: '#f44336', padding: 24 }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 24, color: '#64748b' }}>Loading IoT Devices...</div>
  const { overview: ov, breakdown: bd, definitions: df } = data
  if (!ov.available) return <div style={{ padding: 24 }}>IoT devices config not available.</div>

  const s = ov.summary

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        IoT Devices Fleet Registry
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Emotiv + IoT + Mobile device fleet — connectivity, online/offline strategy, and alert pipeline from iot_devices.json
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

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          <Card title="Fleet KPIs">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, justifyContent: 'center' }}>
              <KPI label="Total Devices" value={s.total_devices} />
              <KPI label="Partial" value={s.partial} sub="integration simulated" />
              <KPI label="Planned" value={s.planned} sub="spec only" />
              <KPI label="Device Types" value={s.device_types} />
              <KPI label="Data Streams" value={s.unique_data_streams} sub="unique" />
              <KPI label="Alert-Capable" value={s.alert_capable} sub="devices with alerts" />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={ov.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {ov.status_distribution.map((_, i) => <Cell key={i} fill={Object.values(STATUS_COLORS)[i] || COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Device Types">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={ov.type_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {ov.type_distribution.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Connectivity Modes">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={ov.mode_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" fontSize={11} />
                  <YAxis allowDecimals={false} fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Device Fleet Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Device</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Type</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Channels</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Modes</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Data Streams</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Alert</th>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.device_summary.map((d, i) => (
                    <tr key={d.id} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{d.name}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{d.type}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{d.channels || '-'}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                        {d.modes.map(m => <ModeBadge key={m} mode={m} />)}
                      </td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{d.data_streams}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                        {d.has_alert ? <span style={{ color: '#4caf50' }}>Yes</span> : <span style={{ color: '#ccc' }}>-</span>}
                      </td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}><StatusBadge status={d.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {s.honest_note && (
            <Card title="Honest Note">
              <p style={{ fontSize: 12, color: '#64748b', margin: 0, fontStyle: 'italic' }}>{s.honest_note}</p>
            </Card>
          )}
        </>
      )}

      {/* BY TYPE TAB */}
      {tab === 'by-type' && bd.available && (
        <>
          {Object.entries(bd.by_type).map(([type, devices]) => (
            <Card key={type} title={`${type} (${devices.length})`}>
              {devices.map(d => (
                <div key={d.id} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{d.name}</span>
                    <StatusBadge status={d.status} />
                  </div>
                  {d.channels && <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Channels: {d.channels}</div>}
                  <div style={{ fontSize: 12, marginBottom: 4 }}>
                    <strong>Modes:</strong> {d.modes.map(m => <ModeBadge key={m} mode={m} />)}
                  </div>
                  <div style={{ fontSize: 12, marginBottom: 4 }}>
                    <strong>Data:</strong> {d.data.map(s => <Pill key={s} text={s} />)}
                  </div>
                  {d.online && <div style={{ fontSize: 11, color: '#475569', marginBottom: 2 }}><strong>Online:</strong> {d.online}</div>}
                  {d.offline && <div style={{ fontSize: 11, color: '#475569', marginBottom: 2 }}><strong>Offline:</strong> {d.offline}</div>}
                  {d.alert && <div style={{ fontSize: 11, color: '#e65100', marginBottom: 2 }}><strong>Alert:</strong> {d.alert}</div>}
                </div>
              ))}
            </Card>
          ))}
        </>
      )}

      {/* DEVICE MATRIX TAB */}
      {tab === 'device-matrix' && bd.available && (
        <>
          <Card title="Mode Matrix (Device x Connectivity)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Device</th>
                    {bd.mode_matrix.modes.map(m => (
                      <th key={m} style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textTransform: 'capitalize' }}>{m}</th>
                    ))}
                    <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.mode_matrix.rows.map((row, i) => (
                    <tr key={row.id} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{row.device}</td>
                      {bd.mode_matrix.modes.map(m => (
                        <td key={m} style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', textAlign: 'center' }}>
                          <BoolBadge val={row[m]} />
                        </td>
                      ))}
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}><StatusBadge status={row.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Data Stream Matrix (Device x Data)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px', borderBottom: '2px solid #e2e8f0', position: 'sticky', left: 0, background: '#f8fafc', zIndex: 1 }}>Device</th>
                    {bd.data_matrix.streams.map(s => (
                      <th key={s} style={{ padding: '6px 8px', borderBottom: '2px solid #e2e8f0', whiteSpace: 'nowrap' }}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bd.data_matrix.rows.map((row, i) => (
                    <tr key={row.id} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9', fontWeight: 600, position: 'sticky', left: 0, background: i % 2 ? '#f8fafc' : '#fff', zIndex: 1 }}>{row.device}</td>
                      {bd.data_matrix.streams.map(s => (
                        <td key={s} style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9', textAlign: 'center' }}>
                          <BoolBadge val={row[s]} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* CONNECTIVITY TAB */}
      {tab === 'connectivity' && (
        <>
          <Card title="Connectivity Model">
            {Object.entries(ov.connectivity_model).map(([mode, desc]) => (
              <div key={mode} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, marginBottom: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <ModeBadge mode={mode} />
                  <span style={{ fontWeight: 600, fontSize: 14, textTransform: 'capitalize' }}>{mode}</span>
                </div>
                <p style={{ fontSize: 12, color: '#475569', margin: 0 }}>{desc}</p>
              </div>
            ))}
          </Card>

          {bd.available && bd.offline_strategy && (
            <Card title="Offline Strategy">
              {Object.entries(bd.offline_strategy).map(([key, desc]) => (
                <div key={key} style={{ marginBottom: 10, padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', textTransform: 'capitalize', marginBottom: 4 }}>
                    {key.replace(/_/g, ' ')}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{desc}</div>
                </div>
              ))}
            </Card>
          )}

          <Card title="Alert Pipeline">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              {ov.alert_pipeline.split(' \u2192 ').map((step, i, arr) => (
                <React.Fragment key={i}>
                  <span style={{
                    background: '#e3f2fd', color: '#1565c0', padding: '6px 14px',
                    borderRadius: 20, fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap'
                  }}>
                    {step}
                  </span>
                  {i < arr.length - 1 && <span style={{ color: '#90a4ae', fontSize: 18 }}>&rarr;</span>}
                </React.Fragment>
              ))}
            </div>
          </Card>
        </>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && df && (
        <>
          <Card title="Device Types">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>Type</th>
                  <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {df.device_types.map((dt, i) => (
                  <tr key={dt.type} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{dt.type}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{dt.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Status Legend">
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {df.status_legend.map(sl => (
                <div key={sl.status} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <StatusBadge status={sl.status} />
                  <span style={{ fontSize: 12, color: '#475569' }}>{sl.description}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', width: 80 }}>Term</th>
                  <th style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {df.glossary.map((g, i) => (
                  <tr key={g.term} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {df.clinical_notes.map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {df.references.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
