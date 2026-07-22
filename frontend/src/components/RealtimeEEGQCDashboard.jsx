import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const VERDICT_COLORS = { Pass: '#22c55e', 'Needs Attention': '#f59e0b', 'Re-record': '#ef4444' }
const SEVERITY_COLORS = { mild: '#22c55e', moderate: '#f59e0b', severe: '#ef4444' }
const BAR_COLORS = ['#3b82f6', '#8b5cf6', '#ef4444', '#f59e0b', '#06b6d4', '#ec4899']

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

function VerdictBadge({ verdict }) {
  const color = VERDICT_COLORS[verdict] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{verdict}</span>
  )
}

export default function RealtimeEEGQCDashboard() {
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
          axios.get(`${API_URL}/api/realtime-eeg-qc/overview`),
          axios.get(`${API_URL}/api/realtime-eeg-qc/breakdown`),
          axios.get(`${API_URL}/api/realtime-eeg-qc/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading EEG QC data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'QC Overview' },
    { id: 'channels', label: 'Channel Analysis' },
    { id: 'artifacts', label: 'Artifact Breakdown' },
    { id: 'recordings', label: 'Recording Log' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Real-Time EEG QC Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Recording quality control — catch problems during acquisition, prevent re-recordings
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#334155', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && renderOverview(overview)}
      {tab === 'channels' && breakdown && renderChannels(breakdown)}
      {tab === 'artifacts' && breakdown && renderArtifacts(breakdown)}
      {tab === 'recordings' && overview && renderRecordings(overview)}
      {tab === 'definitions' && defs && renderDefinitions(defs)}
    </div>
  )
}

function renderOverview(data) {
  const k = data.kpis
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="QC Summary" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <KPI label="Total Recordings" value={fmt(k.total_recordings)} />
          <KPI label="Pass Rate" value={`${fmt(k.pass_rate_pct)}%`} color={k.pass_rate_pct >= 70 ? '#22c55e' : '#f59e0b'} />
          <KPI label="Needs Attention" value={fmt(k.needs_attention)} color="#f59e0b" />
          <KPI label="Re-record" value={fmt(k.re_record)} color="#ef4444" />
        </div>
      </Card>

      <Card title="Channel Quality">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <KPI label="Channels Checked" value={fmt(k.total_channels_checked)} />
          <KPI label="Flagged" value={fmt(k.flagged_channels)} color={k.flagged_channels > 0 ? '#f59e0b' : '#22c55e'} />
          <KPI label="Impedance Flags" value={fmt(k.impedance_flags)} sub="> 10 kΩ" />
          <KPI label="SNR Flags" value={fmt(k.snr_flags)} sub="< 15 dB" />
        </div>
      </Card>

      <Card title="Verdict Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={data.verdict_distribution} dataKey="count" nameKey="verdict" cx="50%" cy="50%"
              outerRadius={70} label={({ verdict, count }) => `${verdict}: ${count}`}>
              {data.verdict_distribution.map((d, i) => (
                <Cell key={i} fill={VERDICT_COLORS[d.verdict] || '#94a3b8'} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {data.alerts.length > 0 && (
        <Card title="Active Alerts" span={2}>
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            {data.alerts.map((a, i) => (
              <div key={i} style={{
                padding: '10px 14px', marginBottom: 8, borderRadius: 8,
                background: a.level === 'critical' ? '#fef2f2' : '#fffbeb',
                borderLeft: `4px solid ${a.level === 'critical' ? '#ef4444' : '#f59e0b'}`
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: a.level === 'critical' ? '#dc2626' : '#d97706' }}>
                  {a.patient_id}
                </div>
                <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>{a.message}</div>
                {a.bad_channels.length > 0 && (
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                    Channels: {a.bad_channels.join(', ')}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function renderChannels(data) {
  const top10 = [...data.channel_summary]
    .sort((a, b) => b.avg_impedance_kohm - a.avg_impedance_kohm)
    .slice(0, 19)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
      <Card title="Impedance by Channel (avg kΩ)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={top10} margin={{ left: 10, right: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="avg_impedance_kohm" name="Avg Impedance (kΩ)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="max_impedance_kohm" name="Max Impedance (kΩ)" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Impedance Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data.impedance_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Channel Detail Table">
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                {['Channel', 'Avg Imp (kΩ)', 'Max Imp (kΩ)', 'Avg SNR (dB)', 'Min SNR (dB)', 'Fail Rate %', 'Recordings'].map(h => (
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.channel_summary.map((ch, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 600 }}>{ch.channel}</td>
                  <td style={{ padding: '6px', color: ch.avg_impedance_kohm > 10 ? '#ef4444' : '#22c55e' }}>{fmt(ch.avg_impedance_kohm)}</td>
                  <td style={{ padding: '6px', color: ch.max_impedance_kohm > 20 ? '#ef4444' : '#475569' }}>{fmt(ch.max_impedance_kohm)}</td>
                  <td style={{ padding: '6px', color: ch.avg_snr_db < 15 ? '#ef4444' : '#22c55e' }}>{fmt(ch.avg_snr_db)}</td>
                  <td style={{ padding: '6px', color: ch.min_snr_db < 8 ? '#ef4444' : '#475569' }}>{fmt(ch.min_snr_db)}</td>
                  <td style={{ padding: '6px', color: ch.impedance_fail_rate_pct > 10 ? '#ef4444' : '#22c55e' }}>{fmt(ch.impedance_fail_rate_pct)}%</td>
                  <td style={{ padding: '6px' }}>{ch.recordings}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderArtifacts(data) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 16 }}>
      <Card title="Artifact Types">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={data.artifact_type_breakdown} dataKey="count" nameKey="type" cx="50%" cy="50%"
              outerRadius={80} label={({ type, count }) => `${type}: ${count}`}>
              {data.artifact_type_breakdown.map((d, i) => (
                <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Artifact Severity">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data.artifact_severity_breakdown}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {data.artifact_severity_breakdown.map((d, i) => (
                <Cell key={i} fill={SEVERITY_COLORS[d.severity] || '#94a3b8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Most Affected Channels (by artifact count)" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data.artifact_by_channel} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="channel" tick={{ fontSize: 11 }} width={50} />
            <Tooltip />
            <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderRecordings(data) {
  return (
    <Card title="Recording QC Log">
      <div style={{ maxHeight: 600, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
              {['Patient', 'Type', 'Duration', 'Rate', 'Montage', 'Date', 'Channels', 'Imp Flags', 'SNR Flags', 'Artifact %', 'Verdict'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.recordings.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: r.verdict === 'Re-record' ? '#fef2f2' : r.verdict === 'Needs Attention' ? '#fffbeb' : undefined }}>
                <td style={{ padding: '6px', fontWeight: 600 }}>{r.patient_id}</td>
                <td style={{ padding: '6px' }}>{r.recording_type}</td>
                <td style={{ padding: '6px' }}>{r.duration_min} min</td>
                <td style={{ padding: '6px' }}>{r.sampling_rate} Hz</td>
                <td style={{ padding: '6px' }}>{r.montage}</td>
                <td style={{ padding: '6px', whiteSpace: 'nowrap' }}>{r.study_date}</td>
                <td style={{ padding: '6px', textAlign: 'center' }}>{r.channel_count}</td>
                <td style={{ padding: '6px', textAlign: 'center', color: (r.impedance_warns + r.impedance_fails) > 0 ? '#ef4444' : '#22c55e' }}>
                  {r.impedance_warns + r.impedance_fails}
                </td>
                <td style={{ padding: '6px', textAlign: 'center', color: (r.snr_warns + r.snr_fails) > 0 ? '#ef4444' : '#22c55e' }}>
                  {r.snr_warns + r.snr_fails}
                </td>
                <td style={{ padding: '6px', textAlign: 'center', color: r.artifact_burden_pct > 20 ? '#ef4444' : '#22c55e' }}>
                  {fmt(r.artifact_burden_pct)}%
                </td>
                <td style={{ padding: '6px' }}><VerdictBadge verdict={r.verdict} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderDefinitions(defs) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 16 }}>
      <Card title="Purpose" span={2}>
        <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{defs.purpose}</p>
      </Card>

      <Card title="Data Sources" span={2}>
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#475569' }}>
          {defs.data_sources.map((s, i) => <li key={i} style={{ marginBottom: 4 }}>{s}</li>)}
        </ul>
      </Card>

      <Card title="QC Checks & Thresholds" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Check', 'Unit', 'Warning', 'Fail', 'Description'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {defs.qc_checks.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 600 }}>{c.check}</td>
                <td style={{ padding: '6px' }}>{c.unit}</td>
                <td style={{ padding: '6px', color: '#f59e0b' }}>{c.warn_threshold}</td>
                <td style={{ padding: '6px', color: '#ef4444' }}>{c.fail_threshold}</td>
                <td style={{ padding: '6px', color: '#64748b', fontSize: 11 }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Verdict Criteria">
        {defs.verdicts.map((v, i) => (
          <div key={i} style={{
            padding: '10px 14px', marginBottom: 8, borderRadius: 8,
            borderLeft: `4px solid ${v.color}`, background: v.color + '11'
          }}>
            <div style={{ fontWeight: 600, fontSize: 13, color: v.color }}>{v.verdict}</div>
            <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>{v.criteria}</div>
          </div>
        ))}
      </Card>

      <Card title="Clinical Reference">
        <p style={{ margin: 0, fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>
          {defs.clinical_reference}
        </p>
      </Card>
    </div>
  )
}
