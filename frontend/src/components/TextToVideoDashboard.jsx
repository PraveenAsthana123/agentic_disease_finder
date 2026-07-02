import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const SEVERITY_COLORS = {
  severe: '#ef4444',
  moderate: '#f59e0b',
  mild: '#10b981',
  unknown: '#94a3b8',
  none: '#e2e8f0',
}

const TIER_COLORS = {
  short: '#10b981',
  medium: '#3b82f6',
  long: '#f59e0b',
  extended: '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'source-inventory', label: 'EEG Sources' },
  { id: 'seizure-clips', label: 'Seizure Clips' },
  { id: 'mri-renders', label: 'MRI Renders' },
  { id: 'patient-profiles', label: 'Patient Profiles' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function TextToVideoDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/text-to-video/overview`),
      axios.get(`${API_URL}/api/text-to-video/breakdown`),
      axios.get(`${API_URL}/api/text-to-video/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Text-to-Video AI data...</div>
  if (err) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {err}</div>
  if (!ov?.available) return <div style={{ padding: 40, color: '#64748b' }}>{ov?.message || 'No data available.'}</div>

  const kpis = ov.kpis || []

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Text-to-Video AI</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Clinical data-to-video synthesis — EEG timelapses, seizure event clips, MRI 3D flythroughs, video annotation pipeline
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#8b5cf6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(kpis, ov)}
      {tab === 'source-inventory' && renderSourceInventory(bd)}
      {tab === 'seizure-clips' && renderSeizureClips(bd)}
      {tab === 'mri-renders' && renderMRIRenders(bd)}
      {tab === 'patient-profiles' && renderPatientProfiles(bd)}
      {tab === 'pipeline' && renderPipeline(bd)}
      {tab === 'definitions' && renderDefinitions(defs)}
    </div>
  )
}

function renderOverview(kpis, ov) {
  return (
    <>
      {/* KPI cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 20 }}>
        {kpis.map((k, i) => (
          <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Video type distribution pie */}
        <Card title="Video Type Distribution">
          {ov.video_type_distribution?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.video_type_distribution} dataKey="count" nameKey="type" cx="50%" cy="50%"
                  outerRadius={80} label={({ type, count }) => `${type}: ${count}`}>
                  {ov.video_type_distribution.map((d, i) => (
                    <Cell key={i} fill={d.color || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        {/* Severity distribution */}
        <Card title="Seizure Severity Distribution">
          {ov.severity_distribution?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.severity_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Events">
                  {ov.severity_distribution.map((d, i) => (
                    <Cell key={i} fill={SEVERITY_COLORS[d.severity] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Confidence distribution */}
        <Card title="Analysis Confidence Distribution">
          {ov.confidence_distribution?.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={ov.confidence_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="tier" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        {/* Daily activity */}
        <Card title="Daily Video Source Activity">
          {ov.daily_activity?.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={ov.daily_activity}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>
      </div>
    </>
  )
}

function renderSourceInventory(bd) {
  const items = bd?.source_inventory || []
  return (
    <Card title={`EEG Source Recordings (${items.length})`} span={2}>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
              {['ID', 'Patient', 'File', 'Disease', 'Prediction', 'Confidence', 'Quality', 'Video Type', 'Duration', 'Date'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}>{s.id}</td>
                <td>{s.patient_id}</td>
                <td style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.file_name}</td>
                <td>{s.disease}</td>
                <td>{s.predicted_label && <Badge text={s.predicted_label} color="#3b82f6" />}</td>
                <td>{s.confidence != null ? (s.confidence * 100).toFixed(1) + '%' : '-'}</td>
                <td>{s.signal_quality || '-'}</td>
                <td><Badge text={s.video_type} color="#8b5cf6" /></td>
                <td><Badge text={s.duration_tier} color={TIER_COLORS[s.duration_tier] || '#6b7280'} /></td>
                <td style={{ fontSize: 11, color: '#94a3b8' }}>{(s.created_at || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderSeizureClips(bd) {
  const items = bd?.seizure_clips || []
  return (
    <Card title={`Seizure Event Clips (${items.length})`} span={2}>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
              {['ID', 'Patient', 'Date', 'Time', 'Duration', 'Clip Length', 'Severity', 'Location', 'Awareness', 'Motor Signs', 'Trigger'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}>{s.id}</td>
                <td>{s.patient_id}</td>
                <td>{s.event_date}</td>
                <td>{s.event_time || '-'}</td>
                <td>{s.duration_sec}s</td>
                <td><Badge text={`${s.clip_duration_sec}s`} color={TIER_COLORS[s.duration_tier] || '#6b7280'} /></td>
                <td><Badge text={s.severity} color={SEVERITY_COLORS[s.severity] || '#94a3b8'} /></td>
                <td>{s.location || '-'}</td>
                <td>{s.awareness || '-'}</td>
                <td>{s.motor_signs || '-'}</td>
                <td style={{ fontSize: 11 }}>{s.trigger || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderMRIRenders(bd) {
  const items = bd?.mri_renders || []
  return (
    <Card title={`MRI 3D Flythrough Renders (${items.length})`} span={2}>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
              {['ID', 'Patient', 'Region', 'Classification', 'Confidence', 'IoU', 'Video Type', 'Duration Tier', 'Date'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}>{m.id}</td>
                <td>{m.patient_id}</td>
                <td>{m.region}</td>
                <td><Badge text={m.classification} color="#8b5cf6" /></td>
                <td>{m.confidence != null ? (typeof m.confidence === 'number' ? (m.confidence * 100).toFixed(1) + '%' : m.confidence) : '-'}</td>
                <td>{m.iou != null ? (typeof m.iou === 'number' ? m.iou.toFixed(3) : m.iou) : '-'}</td>
                <td><Badge text={m.video_type} color="#3b82f6" /></td>
                <td><Badge text={m.duration_tier} color={TIER_COLORS[m.duration_tier] || '#6b7280'} /></td>
                <td style={{ fontSize: 11, color: '#94a3b8' }}>{(m.created_at || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderPatientProfiles(bd) {
  const profiles = bd?.patient_profiles || []
  return (
    <Card title={`Patient Video Profiles (${profiles.length})`} span={2}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
        {profiles.map((p, i) => (
          <div key={i} style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: 14 }}>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>
              {p.name || `Patient ${p.patient_id}`}
            </div>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
              {p.disease} {p.age && `| Age ${p.age}`}
            </div>
            <div style={{ display: 'flex', gap: 12, fontSize: 12, marginBottom: 6 }}>
              <span><strong>{p.total_videos}</strong> videos</span>
              <span><strong>{p.n_uploads}</strong> EEG</span>
              <span><strong>{p.n_seizure_events}</strong> seizures</span>
              <span><strong>{p.n_mri_findings}</strong> MRI</span>
            </div>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
              Est. duration: <strong>{Math.round(p.est_total_duration_sec / 60)}min</strong>
              {p.worst_severity !== 'none' && (
                <span style={{ marginLeft: 8 }}>
                  Worst: <Badge text={p.worst_severity} color={SEVERITY_COLORS[p.worst_severity] || '#94a3b8'} />
                </span>
              )}
            </div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {(p.video_types || []).map((vt, j) => (
                <Badge key={j} text={vt} color="#8b5cf6" />
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

function renderPipeline(bd) {
  const events = bd?.pipeline_events || []
  const actionDist = bd?.action_distribution || []
  return (
    <>
      {actionDist.length > 0 && (
        <Card title="Action Distribution" span={2}>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={actionDist} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="action" tick={{ fontSize: 11 }} width={120} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}
      <Card title={`Pipeline Events (${events.length})`} span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
                {['ID', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Time'].map(h => (
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map((ev, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px' }}>{ev.id}</td>
                  <td>{ev.patient_id || '-'}</td>
                  <td><Badge text={ev.component} color="#06b6d4" /></td>
                  <td>{ev.action}</td>
                  <td>{ev.actor}</td>
                  <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ev.detail}</td>
                  <td style={{ fontSize: 11, color: '#94a3b8' }}>{(ev.ts_utc || '').slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  )
}

function renderDefinitions(defs) {
  if (!defs) return <div style={{ color: '#94a3b8' }}>No definitions available.</div>
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {/* Concepts */}
      <Card title="Video Synthesis & Rendering Concepts">
        <div style={{ display: 'grid', gap: 10 }}>
          {(defs.concepts || []).map((c, i) => (
            <div key={i} style={{ borderLeft: '3px solid #8b5cf6', paddingLeft: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{c.name}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{c.description}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Quality metrics */}
      <Card title="Quality Metrics">
        <div style={{ display: 'grid', gap: 10 }}>
          {(defs.quality_metrics || []).map((m, i) => (
            <div key={i} style={{ borderLeft: '3px solid #10b981', paddingLeft: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{m.name}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{m.description}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Video categories */}
      <Card title="Video Output Categories">
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {(defs.video_categories || []).map((c, i) => (
            <Badge key={i} text={c.label} color="#8b5cf6" />
          ))}
        </div>
      </Card>

      {/* Compliance */}
      <Card title="Compliance & Regulatory">
        <div style={{ display: 'grid', gap: 10 }}>
          {(defs.compliance || []).map((c, i) => (
            <div key={i} style={{ borderLeft: '3px solid #f59e0b', paddingLeft: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{c.ref}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{c.note}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Remediation */}
      <Card title="Remediation Strategies">
        <div style={{ display: 'grid', gap: 10 }}>
          {(defs.remediation || []).map((r, i) => (
            <div key={i} style={{ borderLeft: '3px solid #ef4444', paddingLeft: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>{r.strategy}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{r.description}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
