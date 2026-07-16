import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, ScatterChart, Scatter
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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316']
const GRADE_COLORS = { Good: '#10b981', Fair: '#f59e0b', Poor: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
      fontWeight: 600, background: (color || '#94a3b8') + '22', color: color || '#94a3b8'
    }}>{text}</span>
  )
}

export default function ChannelQualityDashboard() {
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
      axios.get(`${API_URL}/api/channel-quality/overview`),
      axios.get(`${API_URL}/api/channel-quality/breakdown`),
      axios.get(`${API_URL}/api/channel-quality/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Channel Quality data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>EEG Channel Quality Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        EEG channel impedance &amp; signal quality analytics — {overview?.total_recordings} recordings,
        {' '}{overview?.total_channels} channels, {overview?.total_patients} patients
      </p>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const impedancePie = data.impedance_grade_distribution.map((g) => ({ ...g, fill: GRADE_COLORS[g.grade] || '#94a3b8' }))
  const qualityPie = data.quality_grade_distribution.map((g) => ({ ...g, fill: GRADE_COLORS[g.grade] || '#94a3b8' }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      <Card title="Total Recordings"><KPI label="Recordings" value={data.total_recordings} /></Card>
      <Card title="Total Patients"><KPI label="Patients" value={data.total_patients} /></Card>
      <Card title="Total Channels"><KPI label="EEG channels" value={data.total_channels} /></Card>

      <Card title="Avg Impedance"><KPI label="k\u03A9" value={data.avg_impedance_kohm} color="#3b82f6" /></Card>
      <Card title="Avg SNR"><KPI label="dB" value={data.avg_snr_db} color="#8b5cf6" /></Card>
      <Card title="Good Impedance %"><KPI label="% channels" value={`${data.overall_good_impedance_pct}%`} color="#10b981" /></Card>

      <Card title="Good Quality %" span={1}><KPI label="% channels" value={`${data.overall_good_quality_pct}%`} color="#10b981" /></Card>
      <div />
      <div />

      <Card title="Impedance Grade Distribution" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={impedancePie} dataKey="count" nameKey="grade" cx="50%" cy="50%" outerRadius={80} label={({ grade, count }) => `${grade} (${count})`}>
            {impedancePie.map((e, i) => <Cell key={i} fill={e.fill} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Quality Grade Distribution" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={qualityPie} dataKey="count" nameKey="grade" cx="50%" cy="50%" outerRadius={80} label={({ grade, count }) => `${grade} (${count})`}>
            {qualityPie.map((e, i) => <Cell key={i} fill={e.fill} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>
      <div />

      <Card title="Per-Channel Average Impedance (k\u03A9)" span={3}>
        <ResponsiveContainer width="100%" height={Math.max(200, (data.per_channel_avg?.length || 0) * 28)}>
          <BarChart data={data.per_channel_avg} layout="vertical"><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="channel" width={60} tick={{ fontSize: 11 }} /><Tooltip />
            <Bar dataKey="avg_impedance_kohm" fill="#3b82f6" name="Avg Impedance (k\u03A9)" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Per-Channel Average SNR (dB)" span={3}>
        <ResponsiveContainer width="100%" height={Math.max(200, (data.per_channel_avg?.length || 0) * 28)}>
          <BarChart data={data.per_channel_avg} layout="vertical"><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="channel" width={60} tick={{ fontSize: 11 }} /><Tooltip />
            <Bar dataKey="avg_snr_db" fill="#8b5cf6" name="Avg SNR (dB)" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Trend" span={3}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.monthly_trend}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="month" /><YAxis yAxisId="left" /><YAxis yAxisId="right" orientation="right" /><Tooltip />
            <Line yAxisId="left" type="monotone" dataKey="avg_impedance" stroke="#3b82f6" name="Avg Impedance (k\u03A9)" strokeWidth={2} />
            <Line yAxisId="right" type="monotone" dataKey="avg_snr" stroke="#8b5cf6" name="Avg SNR (dB)" strokeWidth={2} />
            <Line yAxisId="left" type="monotone" dataKey="recordings" stroke="#10b981" name="Recordings" strokeWidth={2} />
            <Legend />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  const CHANNELS = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']

  function impedanceColor(v) {
    if (v == null) return '#f8fafc'
    if (v < 5) return '#d1fae5'
    if (v <= 10) return '#fef3c7'
    return '#fee2e2'
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {data.poor_channels_alert && data.poor_channels_alert.length > 0 && (
        <Card title={`Poor Channels Alert (${data.poor_channels_alert.length} patients >30% poor)`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead><tr style={{ background: '#fef2f2' }}>
                <th style={th}>Patient</th><th style={th}>Poor Channels</th><th style={th}>Total</th><th style={th}>% Poor</th>
              </tr></thead>
              <tbody>{data.poor_channels_alert.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={td}>{a.patient_id}</td>
                  <td style={td}><Badge text={a.poor_count} color="#ef4444" /></td>
                  <td style={td}>{a.total}</td>
                  <td style={td}><Badge text={`${a.pct_poor}%`} color="#ef4444" /></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Per-Patient Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>Patient</th><th style={th}>Avg Impedance</th><th style={th}>Avg SNR</th><th style={th}>Good</th><th style={th}>Fair</th><th style={th}>Poor</th><th style={th}>Grade</th><th style={th}>Date</th>
            </tr></thead>
            <tbody>{data.per_patient.map((p, i) => {
              const total = (p.good_channels || 0) + (p.fair_channels || 0) + (p.poor_channels || 0)
              return (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={td}>{p.patient_id}</td>
                  <td style={td}>{p.avg_impedance}</td>
                  <td style={td}>{p.avg_snr}</td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3, minWidth: 40 }}>
                        <div style={{ width: total ? `${(p.good_channels / total * 100)}%` : '0%', height: '100%', background: '#10b981', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{p.good_channels}</span>
                    </div>
                  </td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3, minWidth: 40 }}>
                        <div style={{ width: total ? `${(p.fair_channels / total * 100)}%` : '0%', height: '100%', background: '#f59e0b', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{p.fair_channels}</span>
                    </div>
                  </td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3, minWidth: 40 }}>
                        <div style={{ width: total ? `${(p.poor_channels / total * 100)}%` : '0%', height: '100%', background: '#ef4444', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{p.poor_channels}</span>
                    </div>
                  </td>
                  <td style={td}><Badge text={p.overall_grade} color={GRADE_COLORS[p.overall_grade]} /></td>
                  <td style={td}>{p.created_at?.slice(0, 10)}</td>
                </tr>
              )
            })}</tbody>
          </table>
        </div>
      </Card>

      <Card title="Channel Impedance Heatmap" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>Patient</th>
              {CHANNELS.map(ch => <th key={ch} style={{ ...th, textAlign: 'center', padding: '6px 4px' }}>{ch}</th>)}
            </tr></thead>
            <tbody>{data.channel_impedance_heatmap?.map((row, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}>{row.patient_id}</td>
                {CHANNELS.map(ch => (
                  <td key={ch} style={{ ...td, textAlign: 'center', padding: '4px 2px', background: impedanceColor(row[ch]), fontWeight: 500 }}>
                    {row[ch] != null ? row[ch] : '—'}
                  </td>
                ))}
              </tr>
            ))}</tbody>
          </table>
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 12, fontSize: 11, color: '#64748b' }}>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#d1fae5', borderRadius: 2, marginRight: 4, verticalAlign: 'middle' }} />{'<5 k\u03A9 (Good)'}</span>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#fef3c7', borderRadius: 2, marginRight: 4, verticalAlign: 'middle' }} />{'5\u201310 k\u03A9 (Fair)'}</span>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#fee2e2', borderRadius: 2, marginRight: 4, verticalAlign: 'middle' }} />{'>10 k\u03A9 (Poor)'}</span>
        </div>
      </Card>

      <Card title="Impedance vs SNR (All Channels)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart><CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="impedance_kohm" name="Impedance (k\u03A9)" unit=" k\u03A9" />
            <YAxis type="number" dataKey="snr_db" name="SNR (dB)" unit=" dB" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => {
              if (!payload || !payload.length) return null
              const d = payload[0].payload
              return (
                <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, padding: 8, fontSize: 12 }}>
                  <div><strong>{d.channel}</strong> (Patient {d.patient_id})</div>
                  <div>Impedance: {d.impedance_kohm} k&#937;</div>
                  <div>SNR: {d.snr_db} dB</div>
                </div>
              )
            }} />
            <Scatter data={data.impedance_vs_snr} fill="#3b82f6" fillOpacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

const th = { textAlign: 'left', padding: '8px 10px', fontSize: 11, color: '#64748b', fontWeight: 600 }
const td = { padding: '8px 10px' }

function DefinitionsTab({ data }) {
  const sections = [
    { title: 'Field Definitions', items: data.fields },
    { title: 'Impedance Grades', items: data.impedance_grades },
    { title: 'Quality Grades', items: data.quality_grades },
    { title: 'Channel Positions', items: data.channels },
    { title: 'Clinical Glossary', items: data.glossary },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map((sec, i) => (
        <Card key={i} title={sec.title} span={sec.items && Object.keys(sec.items).length > 6 ? 2 : 1}>
          {sec.items && Object.entries(sec.items).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 12, color: '#334155' }}>{k}: </span>
              <span style={{ fontSize: 12, color: '#64748b' }}>{v}</span>
            </div>
          ))}
        </Card>
      ))}

      {data.clinical_notes && (
        <Card title="Clinical Notes" span={2}>
          {Array.isArray(data.clinical_notes)
            ? data.clinical_notes.map((note, i) => (
                <div key={i} style={{ marginBottom: 8, fontSize: 12, color: '#64748b', paddingLeft: 12, borderLeft: '3px solid #e2e8f0' }}>
                  {note}
                </div>
              ))
            : Object.entries(data.clinical_notes).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 12, color: '#334155' }}>{k}: </span>
                  <span style={{ fontSize: 12, color: '#64748b' }}>{v}</span>
                </div>
              ))
          }
        </Card>
      )}
    </div>
  )
}
