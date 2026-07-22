import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const SEVERITY_COLORS = { mild: '#22c55e', moderate: '#f59e0b', severe: '#ef4444' }
const TYPE_COLORS = ['#3b82f6', '#8b5cf6', '#ef4444', '#f59e0b', '#06b6d4', '#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

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

function SeverityBadge({ severity }) {
  const color = SEVERITY_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{severity}</span>
  )
}

export default function ArtifactDetectionDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [selectedPatient, setSelectedPatient] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/artifact-detection/overview`),
          axios.get(`${API_URL}/api/artifact-detection/breakdown`),
          axios.get(`${API_URL}/api/artifact-detection/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading artifact detection data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Per-Patient' },
    { id: 'channels', label: 'Channel Quality' },
    { id: 'flagged', label: 'Flagged' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Artifact Detection Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Automated EEG artifact detection and channel quality analysis — reduces manual artifact marking burden
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'patients' && <PatientsTab data={breakdown} selected={selectedPatient} onSelect={setSelectedPatient} />}
      {tab === 'channels' && <ChannelsTab data={breakdown} />}
      {tab === 'flagged' && <FlaggedTab data={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}


function OverviewTab({ data }) {
  if (!data) return null
  const k = data.kpis

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      <Card span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
          <KPI label="Total Artifacts" value={fmt(k.total_artifacts)} />
          <KPI label="Patients Scanned" value={fmt(k.patients_scanned)} />
          <KPI label="Severe Artifacts" value={fmt(k.severe_artifacts)} color="#ef4444" />
          <KPI label="Avg per Patient" value={fmt(k.avg_per_patient)} />
          <KPI label="Avg SNR (dB)" value={fmt(k.avg_snr_db)} sub="across all channels" />
          <KPI label="Poor Channels" value={`${fmt(k.poor_channel_pct)}%`} color={k.poor_channel_pct > 10 ? '#ef4444' : '#22c55e'} />
        </div>
      </Card>

      <Card title="Artifact Type Distribution">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={data.type_distribution} dataKey="count" nameKey="label" cx="50%" cy="50%" outerRadius={80} label={({ label, count }) => `${label}: ${count}`}>
              {data.type_distribution.map((_, i) => <Cell key={i} fill={TYPE_COLORS[i % TYPE_COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Severity Breakdown">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.severity_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="severity" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" radius={[6,6,0,0]}>
              {data.severity_distribution.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Channel Hotspots (Top 10)">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.channel_hotspots} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="channel" type="category" width={40} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0,6,6,0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Type x Severity Matrix" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Artifact Type</th>
                <th style={{ padding: 8, color: SEVERITY_COLORS.mild }}>Mild</th>
                <th style={{ padding: 8, color: SEVERITY_COLORS.moderate }}>Moderate</th>
                <th style={{ padding: 8, color: SEVERITY_COLORS.severe }}>Severe</th>
              </tr>
            </thead>
            <tbody>
              {data.type_severity_matrix.map(r => (
                <tr key={r.type} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 500 }}>{r.label}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>{r.mild}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>{r.moderate}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>{r.severe}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Artifact Rate (per recording minute)">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.artifact_rate.slice(0, 10)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip formatter={(v) => v.toFixed(3)} />
            <Bar dataKey="rate_per_min" fill="#8b5cf6" radius={[6,6,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}


function PatientsTab({ data, selected, onSelect }) {
  if (!data) return null
  const patients = data.patients

  if (selected) {
    const p = patients.find(x => x.patient_id === selected)
    if (!p) return <div>Patient not found</div>
    return (
      <div>
        <button onClick={() => onSelect(null)} style={{
          padding: '6px 14px', borderRadius: 8, border: '1px solid #e2e8f0', background: '#fff',
          cursor: 'pointer', fontSize: 13, marginBottom: 16
        }}>Back to list</button>

        <Card title={`${p.name} (${p.patient_id})`} span={3}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
            <KPI label="Total Artifacts" value={p.total_artifacts} />
            <KPI label="Severe" value={p.severe_count} color="#ef4444" />
            <KPI label="Dominant Type" value={p.dominant_artifact} />
            <KPI label="Recording" value={`${p.recording.duration_min} min`} sub={`${p.recording.sampling_rate} Hz ${p.recording.montage}`} />
          </div>

          <h4 style={{ fontSize: 14, color: '#334155', margin: '16px 0 8px' }}>Channel Quality</h4>
          <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
            <span style={{ fontSize: 13 }}>Good: <b style={{ color: '#22c55e' }}>{p.channel_quality.good}</b></span>
            <span style={{ fontSize: 13 }}>Fair: <b style={{ color: '#f59e0b' }}>{p.channel_quality.fair}</b></span>
            <span style={{ fontSize: 13 }}>Poor: <b style={{ color: '#ef4444' }}>{p.channel_quality.poor}</b></span>
            <span style={{ fontSize: 13 }}>Avg Impedance: <b>{p.channel_quality.avg_impedance_kohm} kOhm</b></span>
          </div>

          <h4 style={{ fontSize: 14, color: '#334155', margin: '16px 0 8px' }}>Artifact Timeline</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>Time (min)</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Type</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Channel</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Duration (s)</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Severity</th>
                </tr>
              </thead>
              <tbody>
                {p.artifact_list.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6 }}>{fmt(a.start_min)}</td>
                    <td style={{ padding: 6 }}>{a.type}</td>
                    <td style={{ padding: 6, fontFamily: 'monospace' }}>{a.channel}</td>
                    <td style={{ padding: 6 }}>{fmt(a.duration_sec)}</td>
                    <td style={{ padding: 6 }}><SeverityBadge severity={a.severity} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <Card title="All Patients">
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
            <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
            <th style={{ padding: 8 }}>Artifacts</th>
            <th style={{ padding: 8 }}>Severe</th>
            <th style={{ padding: 8 }}>Dominant Type</th>
            <th style={{ padding: 8 }}>Channel Quality</th>
            <th style={{ padding: 8 }}>Recording</th>
            <th style={{ padding: 8 }}></th>
          </tr>
        </thead>
        <tbody>
          {patients.map(p => (
            <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: 8 }}>{p.name}<br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{p.patient_id}</span></td>
              <td style={{ padding: 8, textAlign: 'center' }}>{p.total_artifacts}</td>
              <td style={{ padding: 8, textAlign: 'center' }}>{p.severe_count > 0 ? <span style={{ color: '#ef4444', fontWeight: 600 }}>{p.severe_count}</span> : '0'}</td>
              <td style={{ padding: 8, textAlign: 'center' }}>{p.dominant_artifact}</td>
              <td style={{ padding: 8, textAlign: 'center' }}>
                <span style={{ color: '#22c55e' }}>{p.channel_quality.good}G</span>{' / '}
                <span style={{ color: '#f59e0b' }}>{p.channel_quality.fair}F</span>{' / '}
                <span style={{ color: '#ef4444' }}>{p.channel_quality.poor}P</span>
              </td>
              <td style={{ padding: 8, textAlign: 'center', fontSize: 11 }}>{p.recording.duration_min}m @ {p.recording.sampling_rate}Hz</td>
              <td style={{ padding: 8 }}>
                <button onClick={() => onSelect(p.patient_id)} style={{
                  padding: '4px 10px', borderRadius: 6, border: '1px solid #e2e8f0',
                  background: '#f8fafc', cursor: 'pointer', fontSize: 12
                }}>View</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}


function ChannelsTab({ data }) {
  if (!data) return null
  const channels = data.channel_summary

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Channel Quality Summary" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Channel</th>
              <th style={{ padding: 8 }}>Avg SNR (dB)</th>
              <th style={{ padding: 8 }}>Avg Impedance (kOhm)</th>
              <th style={{ padding: 8 }}>Poor Rate (%)</th>
              <th style={{ padding: 8 }}>Recordings</th>
            </tr>
          </thead>
          <tbody>
            {channels.map(ch => (
              <tr key={ch.channel} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontFamily: 'monospace', fontWeight: 600 }}>{ch.channel}</td>
                <td style={{ padding: 8, textAlign: 'center', color: ch.avg_snr_db < 15 ? '#ef4444' : ch.avg_snr_db < 20 ? '#f59e0b' : '#22c55e' }}>
                  {ch.avg_snr_db}
                </td>
                <td style={{ padding: 8, textAlign: 'center', color: ch.avg_impedance_kohm > 10 ? '#ef4444' : ch.avg_impedance_kohm > 5 ? '#f59e0b' : '#22c55e' }}>
                  {ch.avg_impedance_kohm}
                </td>
                <td style={{ padding: 8, textAlign: 'center' }}>
                  <span style={{ color: ch.poor_rate_pct > 20 ? '#ef4444' : '#64748b' }}>{ch.poor_rate_pct}%</span>
                </td>
                <td style={{ padding: 8, textAlign: 'center' }}>{ch.n_recordings}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="SNR by Channel">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={channels}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
            <YAxis label={{ value: 'dB', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="avg_snr_db" fill="#3b82f6" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Impedance by Channel">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={channels}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" tick={{ fontSize: 10 }} />
            <YAxis label={{ value: 'kOhm', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="avg_impedance_kohm" radius={[4,4,0,0]}>
              {channels.map((ch, i) => (
                <Cell key={i} fill={ch.avg_impedance_kohm > 10 ? '#ef4444' : ch.avg_impedance_kohm > 5 ? '#f59e0b' : '#22c55e'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}


function FlaggedTab({ data }) {
  if (!data) return null
  const flagged = data.flagged_patients

  return (
    <Card title={`Flagged Patients (${flagged.length})`}>
      <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 12px' }}>
        Patients flagged for review: 2+ severe artifacts, 8+ total artifacts, or 5+ poor-quality channels
      </p>
      {flagged.length === 0 ? (
        <p style={{ color: '#22c55e', fontWeight: 500 }}>No patients flagged</p>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ padding: 8 }}>Reason</th>
              <th style={{ padding: 8 }}>Total Artifacts</th>
              <th style={{ padding: 8 }}>Severe</th>
              <th style={{ padding: 8 }}>Poor Channels</th>
            </tr>
          </thead>
          <tbody>
            {flagged.map(f => (
              <tr key={f.patient_id} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                <td style={{ padding: 8 }}>{f.name}<br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{f.patient_id}</span></td>
                <td style={{ padding: 8, textAlign: 'center', color: '#ef4444', fontWeight: 500 }}>{f.reason}</td>
                <td style={{ padding: 8, textAlign: 'center' }}>{f.total_artifacts}</td>
                <td style={{ padding: 8, textAlign: 'center' }}>{f.severe_count}</td>
                <td style={{ padding: 8, textAlign: 'center' }}>{f.poor_channels}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  )
}


function DefinitionsTab({ data }) {
  if (!data) return null

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Artifact Types" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {data.artifact_types.map(at => (
            <div key={at.type} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{at.label}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>{at.description}</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                Channels: {Array.isArray(at.typical_channels) ? at.typical_channels.join(', ') : at.typical_channels} | Freq: {at.frequency_range}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Severity Scale">
        {data.severity_scale.map(s => (
          <div key={s.level} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
            <SeverityBadge severity={s.level} />
            <span style={{ fontSize: 13, color: '#475569' }}>{s.description}</span>
          </div>
        ))}
      </Card>

      <Card title="Quality Grades">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Grade</th>
              <th style={{ padding: 6 }}>Impedance</th>
              <th style={{ padding: 6 }}>SNR</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {data.quality_grades.map(g => (
              <tr key={g.grade} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6, fontWeight: 600 }}>{g.grade}</td>
                <td style={{ padding: 6, textAlign: 'center' }}>{g.impedance}</td>
                <td style={{ padding: 6, textAlign: 'center' }}>{g.snr}</td>
                <td style={{ padding: 6 }}>{g.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Glossary" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {data.glossary.map(g => (
            <div key={g.term} style={{ padding: 8, background: '#f8fafc', borderRadius: 6 }}>
              <span style={{ fontWeight: 600, fontSize: 13 }}>{g.term}:</span>{' '}
              <span style={{ fontSize: 12, color: '#475569' }}>{g.definition}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
