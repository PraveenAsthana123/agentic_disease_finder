import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
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

function StatusBadge({ status }) {
  const map = {
    ready: { bg: '#dcfce7', fg: '#166534' },
    needs_quality_check: { bg: '#fef3c7', fg: '#92400e' },
    specification_ready: { bg: '#dbeafe', fg: '#1e40af' },
    planned: { bg: '#f1f5f9', fg: '#475569' },
  }
  const s = map[status] || { bg: '#f1f5f9', fg: '#475569' }
  return <Badge text={status} bg={s.bg} fg={s.fg} />
}

function QualityBadge({ grade }) {
  const map = {
    Good: { bg: '#dcfce7', fg: '#166534' },
    Fair: { bg: '#fef3c7', fg: '#92400e' },
    Poor: { bg: '#fee2e2', fg: '#991b1b' },
  }
  const s = map[grade] || { bg: '#f1f5f9', fg: '#475569' }
  return <Badge text={grade} bg={s.bg} fg={s.fg} />
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

const TABS = ['Overview', 'Channels & Quality', 'Artifacts', 'Recordings', 'Reference']

export default function SegmentationDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/segmentation/overview`),
      axios.get(`${API}/api/segmentation/breakdown`),
      axios.get(`${API}/api/segmentation/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading segmentation data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px' }}>
        EEG Waveform Segmentation Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 20px' }}>
        Waveform digitization pipeline readiness, channel quality, artifact impact, and segmentation methods
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

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <ChannelsTab bd={bd} />}
      {tab === 2 && <ArtifactsTab ov={ov} bd={bd} />}
      {tab === 3 && <RecordingsTab bd={bd} />}
      {tab === 4 && <ReferenceTab df={df} />}
    </div>
  )
}

/* ── Tab 1: Overview ── */
function OverviewTab({ ov }) {
  if (!ov) return null
  const s = ov.summary || {}
  const qualDist = ov.quality_distribution || {}
  const qualData = Object.entries(qualDist).map(([k, v]) => ({ name: k, value: v }))
  const impDist = ov.impedance_distribution || {}
  const impData = Object.entries(impDist).map(([k, v]) => ({ name: k, value: v }))
  const methods = ov.methods || []
  const recTypes = ov.recording_types || []
  const montages = ov.montages || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Pipeline Readiness" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <KPI label="Total Recordings" value={fmt(s.total_recordings)} color="#3b82f6" />
          <KPI label="Ready Patients" value={fmt(s.segmentation_ready_patients)} sub={`${fmt(s.readiness_rate_pct)}% readiness`} color="#10b981" />
          <KPI label="Good Quality Rate" value={`${fmt(s.good_quality_rate_pct)}%`} color="#8b5cf6" />
          <KPI label="Avg SNR" value={`${fmt(s.avg_snr_db)} dB`} color="#06b6d4" />
        </div>
      </Card>

      <Card title="Channel Quality Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={qualData} dataKey="value" nameKey="name"
              cx="50%" cy="50%" outerRadius={90}
              label={({ name, value }) => `${name}: ${value}`}>
              {qualData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Impedance Grade Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={impData} dataKey="value" nameKey="name"
              cx="50%" cy="50%" outerRadius={90}
              label={({ name, value }) => `${name}: ${value}`}>
              {impData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Recording Types" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={recTypes}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Recordings" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Montage Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={montages}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="montage" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Recordings" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Segmentation Methods" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Method', 'Approach', 'Input', 'Output', 'Pros', 'Cons', 'Status'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {methods.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{m.name}</td>
                  <td style={{ padding: '8px 10px', color: '#475569', maxWidth: 200 }}>{m.approach}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12 }}>{m.input}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12 }}>{m.output}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12, color: '#166534' }}>{m.pros}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12, color: '#991b1b' }}>{m.cons}</td>
                  <td style={{ padding: '8px 10px' }}><StatusBadge status={m.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ── Tab 2: Channels & Quality ── */
function ChannelsTab({ bd }) {
  if (!bd) return null
  const channels = bd.channel_summary || []
  const srDist = bd.sampling_rate_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Channel Quality KPIs" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <KPI label="Channels Assessed" value={fmt(channels.length)} color="#3b82f6" />
          <KPI label="Avg Good %" value={channels.length ? `${(channels.reduce((a, c) => a + c.good_pct, 0) / channels.length).toFixed(1)}%` : '--'} color="#10b981" />
          <KPI label="Avg SNR (dB)" value={channels.length ? (channels.reduce((a, c) => a + c.avg_snr_db, 0) / channels.length).toFixed(1) : '--'} color="#06b6d4" />
        </div>
      </Card>

      <Card title="Per-Channel SNR (dB)" span={4}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={channels}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
            <YAxis label={{ value: 'SNR (dB)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <Tooltip />
            <Bar dataKey="avg_snr_db" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Avg SNR (dB)" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Per-Channel Impedance (kohm)" span={4}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={channels}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
            <YAxis label={{ value: 'Impedance (kohm)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <Tooltip />
            <Bar dataKey="avg_impedance_kohm" radius={[4, 4, 0, 0]} name="Avg Impedance">
              {channels.map((c, i) => <Cell key={i} fill={c.avg_impedance_kohm > 10 ? '#ef4444' : c.avg_impedance_kohm > 5 ? '#f59e0b' : '#10b981'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Channel Quality Detail" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Channel', 'Recordings', 'Avg Impedance', 'Avg SNR', 'Good %', 'Fair %', 'Poor %'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {channels.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.channel}</td>
                  <td style={{ padding: '8px 10px' }}>{c.recordings}</td>
                  <td style={{ padding: '8px 10px' }}>{c.avg_impedance_kohm} kohm</td>
                  <td style={{ padding: '8px 10px' }}>{c.avg_snr_db} dB</td>
                  <td style={{ padding: '8px 10px', color: '#166534', fontWeight: 600 }}>{c.good_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#92400e' }}>{c.fair_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#991b1b' }}>{c.poor_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Sampling Rate Distribution" span={4}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={srDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="rate" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Recordings" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* ── Tab 3: Artifacts ── */
function ArtifactsTab({ ov, bd }) {
  if (!ov) return null
  const s = ov.summary || {}
  const artTypes = ov.artifact_types || []
  const sevDist = ov.artifact_severity || {}
  const sevData = Object.entries(sevDist).map(([k, v]) => ({ name: k, value: v }))
  const artChannels = (bd && bd.artifact_by_channel) || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Artifact Impact KPIs" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <KPI label="Total Artifacts" value={fmt(s.total_artifacts)} color="#ef4444" />
          <KPI label="Avg Duration" value={`${fmt(s.avg_artifact_duration_sec)}s`} color="#f59e0b" />
          <KPI label="Distinct Patients" value={fmt(s.distinct_patients)} color="#3b82f6" />
        </div>
      </Card>

      <Card title="Artifact Types" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={artTypes}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} name="Count" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Severity Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={sevData} dataKey="value" nameKey="name"
              cx="50%" cy="50%" outerRadius={90}
              label={({ name, value }) => `${name}: ${value}`}>
              {sevData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Artifacts by Channel" span={4}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={artChannels}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="total_artifacts" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Artifacts" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Artifact Detail by Channel" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Channel', 'Total', 'Types', 'Mild', 'Moderate', 'Severe'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {artChannels.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.channel}</td>
                  <td style={{ padding: '8px 10px' }}>{c.total_artifacts}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12 }}>{Object.entries(c.types || {}).map(([k, v]) => `${k}(${v})`).join(', ')}</td>
                  <td style={{ padding: '8px 10px', color: '#166534' }}>{(c.severities || {}).mild || 0}</td>
                  <td style={{ padding: '8px 10px', color: '#92400e' }}>{(c.severities || {}).moderate || 0}</td>
                  <td style={{ padding: '8px 10px', color: '#991b1b' }}>{(c.severities || {}).severe || 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ── Tab 4: Recordings ── */
function RecordingsTab({ bd }) {
  if (!bd) return null
  const recordings = bd.patient_recordings || []
  const readiness = bd.patient_readiness || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Recording KPIs" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <KPI label="Total Recordings" value={fmt(recordings.length)} color="#3b82f6" />
          <KPI label="Segmentation Ready" value={fmt(readiness.filter(r => r.segmentation_status === 'ready').length)} color="#10b981" />
          <KPI label="Need Quality Check" value={fmt(readiness.filter(r => r.segmentation_status === 'needs_quality_check').length)} color="#f59e0b" />
        </div>
      </Card>

      <Card title="Patient Segmentation Readiness" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Patient', 'Acquisition', 'Channel Quality', 'Artifacts', 'Status'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {readiness.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '8px 10px' }}><QualityBadge grade={r.has_acquisition ? 'Good' : 'Poor'} /></td>
                  <td style={{ padding: '8px 10px' }}><QualityBadge grade={r.has_channel_quality ? 'Good' : 'Poor'} /></td>
                  <td style={{ padding: '8px 10px' }}>{r.has_artifact_annotations ? <Badge text="annotated" bg="#dbeafe" fg="#1e40af" /> : <Badge text="none" bg="#f1f5f9" fg="#475569" />}</td>
                  <td style={{ padding: '8px 10px' }}><StatusBadge status={r.segmentation_status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title={`EEG Recordings Detail (${recordings.length})`} span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Patient', 'Type', 'Duration (min)', 'Sample Rate', 'Montage', 'Electrode', 'Study Date', 'Notes'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {recordings.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '8px 10px' }}>{r.recording_type}</td>
                  <td style={{ padding: '8px 10px' }}>{r.duration_min}</td>
                  <td style={{ padding: '8px 10px' }}>{r.sampling_rate} Hz</td>
                  <td style={{ padding: '8px 10px' }}>{r.montage}</td>
                  <td style={{ padding: '8px 10px' }}>{r.electrode_system}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12 }}>{r.study_date || '--'}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.technician_notes || '--'}</td>
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
  const defs = df.definitions || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Metric Definitions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Metric', 'Definition'].map(h =>
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {defs.map((d, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>{d.metric}</td>
                <td style={{ padding: '8px 10px', color: '#475569' }}>{d.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
