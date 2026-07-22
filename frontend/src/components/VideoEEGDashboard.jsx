import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const SEV_COLORS = { Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const OUTCOME_COLORS = ['#16a34a', '#3b82f6', '#94a3b8']
const HOUR_COLORS = { 'Night (00-06)': '#6366f1', 'Morning (06-12)': '#f59e0b', 'Afternoon (12-18)': '#10b981', 'Evening (18-24)': '#8b5cf6' }

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
  const color = SEV_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{severity || 'Unknown'}</span>
  )
}

export default function VideoEEGDashboard() {
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
          axios.get(`${API_URL}/api/video-eeg/overview`),
          axios.get(`${API_URL}/api/video-eeg/breakdown`),
          axios.get(`${API_URL}/api/video-eeg/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Video EEG data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'seizures', label: 'Seizure Analysis' },
    { id: 'eeg', label: 'EEG Features & Concordance' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const monDist = overview?.monitoring_distribution || []
  const sevDist = overview?.seizure_severity_distribution || []
  const patients = overview?.per_patient_summary || []
  const timeline = breakdown?.seizure_timeline || []
  const durHist = breakdown?.duration_histogram || []
  const auraDist = breakdown?.aura_distribution || []
  const triggers = breakdown?.trigger_analysis || []
  const temporal = breakdown?.temporal_pattern || []
  const eegFeatures = breakdown?.eeg_features || []
  const concordance = breakdown?.concordance || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Video EEG Monitoring Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Continuous EEG + Video — Seizure Capture, Classification & Pre-Surgical Evaluation
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Sessions" value={fmt(kpis.total_sessions)} />
              <KPI label="Seizures Captured" value={fmt(kpis.total_seizures_captured)} />
              <KPI label="Mean Duration" value={`${fmt(kpis.mean_duration_sec)}s`} sub="seconds" />
              <KPI label="Ictal Capture Rate" value={`${fmt(kpis.ictal_capture_rate)}%`} color={kpis.ictal_capture_rate >= 80 ? '#16a34a' : '#f59e0b'} />
              <KPI label="Patients Monitored" value={fmt(kpis.patients_monitored)} />
            </div>
          </Card>

          {/* Monitoring Outcome Distribution */}
          <Card title="Monitoring Outcomes">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={monDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={80} paddingAngle={2}>
                  {monDist.map((d, i) => <Cell key={i} fill={OUTCOME_COLORS[i % OUTCOME_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {monDist.map((d, i) => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: OUTCOME_COLORS[i % OUTCOME_COLORS.length], marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Severity Distribution */}
          <Card title="Seizure Severity" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={sevDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" name="Count">
                  {sevDist.map((d, i) => <Cell key={i} fill={SEV_COLORS[d.name] || '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Patient Summary Table */}
          <Card title="Per-Patient Monitoring Summary" span={3}>
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Sessions</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Seizures</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Mean Dur (s)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Longest (s)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Aura</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Awareness</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.sessions}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.seizures_captured}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.mean_duration_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.longest_event_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.had_aura ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.awareness_level || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><SeverityBadge severity={p.severity} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── SEIZURE ANALYSIS TAB ── */}
      {tab === 'seizures' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Duration Histogram */}
          <Card title="Seizure Duration Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={durHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" name="Events" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Temporal Pattern */}
          <Card title="Seizure Temporal Pattern">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={temporal}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour_block" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" name="Events">
                  {temporal.map((d, i) => <Cell key={i} fill={HOUR_COLORS[d.hour_block] || '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Aura Distribution */}
          <Card title="Aura Types">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={auraDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={35} outerRadius={75} paddingAngle={2}>
                  {auraDist.map((d, i) => <Cell key={i} fill={['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#94a3b8'][i % 6]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {auraDist.map((d, i) => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#94a3b8'][i % 6], marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Trigger Analysis */}
          <Card title="Seizure Triggers">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={triggers} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis dataKey="trigger" type="category" tick={{ fontSize: 10 }} width={120} />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" name="Events" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Seizure Event Timeline Table */}
          <Card title="Seizure Event Log" span={2}>
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Date</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Time</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Duration (s)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Severity</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Aura</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Awareness</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Motor Signs</th>
                  </tr>
                </thead>
                <tbody>
                  {timeline.map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{e.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{e.event_date || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{e.event_time || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(e.duration_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><SeverityBadge severity={e.severity} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{e.aura || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{e.awareness || '--'}</td>
                      <td style={{ padding: '6px 8px' }}>{e.motor_signs || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── EEG FEATURES & CONCORDANCE TAB ── */}
      {tab === 'eeg' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)', gap: 16 }}>
          {/* EEG Feature Table */}
          <Card title="CHB-MIT EEG Seizure Features (PhysioNet)">
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Subject</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>File</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Onset (s)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Offset (s)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Duration (s)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Channels</th>
                  </tr>
                </thead>
                <tbody>
                  {eegFeatures.map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{f.patient_id}</td>
                      <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>{f.file}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(f.onset_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(f.offset_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(f.duration_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{f.channels_involved}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Concordance Table */}
          <Card title="Clinical-EEG Concordance (Seizure Diary vs EEG Captures)">
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Clinical Seizures</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>EEG Seizures</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Concordant</th>
                  </tr>
                </thead>
                <tbody>
                  {concordance.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{c.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{c.clinical_seizures}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{c.eeg_seizures}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                          background: c.concordant ? '#dcfce7' : '#fef9c3',
                          color: c.concordant ? '#166534' : '#854d0e'
                        }}>{c.concordant ? 'Yes' : 'No'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Monitoring Protocol */}
          {defs.monitoring_protocol && (
            <Card title="Standard Video-EEG Monitoring Protocol">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.monitoring_protocol).map(([k, v]) => (
                  <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: '#64748b', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 13, color: '#1e293b', marginTop: 2 }}>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Seizure Semiology */}
          {defs.seizure_semiology && defs.seizure_semiology.length > 0 && (
            <Card title="Seizure Semiology Reference">
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', width: 160 }}>Term</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.seizure_semiology.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{s.term}</td>
                      <td style={{ padding: '6px 8px' }}>{s.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Metric Definitions */}
          <Card title="Metric Definitions">
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', width: 180 }}>Metric</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', width: 120 }}>Source</th>
                </tr>
              </thead>
              <tbody>
                {(defs.metrics || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '6px 8px' }}>{m.description}</td>
                    <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{m.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Clinical Significance */}
          {defs.clinical_significance && (
            <Card title="Clinical Significance">
              <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6, margin: 0 }}>{defs.clinical_significance}</p>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
