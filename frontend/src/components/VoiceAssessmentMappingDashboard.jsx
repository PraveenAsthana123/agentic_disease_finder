import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
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

const TABS = ['Overview', 'Sessions', 'Patients', 'Instruments', 'Definitions']
const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6']
const STATUS_COLORS = { Completed: '#10b981', 'In Progress': '#3b82f6', Abandoned: '#ef4444' }

export default function VoiceAssessmentMappingDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/api/voice-assessment-mapping/overview`).then(r => setOv(r.data)).catch(e => setErr(e.message))
    axios.get(`${API_URL}/api/voice-assessment-mapping/breakdown`).then(r => setBd(r.data)).catch(() => {})
    axios.get(`${API_URL}/api/voice-assessment-mapping/definitions`).then(r => setDefs(r.data)).catch(() => {})
  }, [])

  if (err) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {err}</div>
  if (!ov) return <div style={{ padding: 32, color: '#64748b' }}>Loading Voice Assessment Mapping...</div>
  if (ov.available === false) return <div style={{ padding: 32, color: '#64748b' }}>{ov.message}</div>

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22 }}>Voice Assessment Mapping</h2>
      <p style={{ margin: '0 0 16px', color: '#64748b', fontSize: 13 }}>
        STT-to-form pipeline — maps voice input to structured clinical assessments
      </p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#6366f1' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <SessionsTab bd={bd} />}
      {tab === 2 && <PatientsTab bd={bd} />}
      {tab === 3 && <InstrumentsTab bd={bd} ov={ov} />}
      {tab === 4 && <DefinitionsTab defs={defs} />}
    </div>
  )
}

/* ── Overview Tab ─────────────────────────────────────────────────── */
function OverviewTab({ ov }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {(ov.kpis || []).map(k => (
        <Card key={k.label}><KPI label={k.label} value={k.value} color={k.color} /></Card>
      ))}

      <Card title="Channel Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={ov.channel_distribution || []} dataKey="count" nameKey="channel"
                 cx="50%" cy="50%" outerRadius={85} label={({ channel, count }) => `${channel}: ${count}`}>
              {(ov.channel_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Session Status" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={ov.status_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="status" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count">
              {(ov.status_distribution || []).map((d, i) => (
                <Cell key={i} fill={STATUS_COLORS[d.status] || COLORS[i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Instrument Coverage by Channel" span={4}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={ov.instrument_coverage || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="instrument" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="voice" name="Voice AI" fill="#6366f1" />
            <Bar dataKey="chat" name="Conversational AI" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {(ov.daily_activity || []).length > 0 && (
        <Card title="Daily Voice Mapping Activity" span={4}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={ov.daily_activity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}
    </div>
  )
}

/* ── Sessions Tab ─────────────────────────────────────────────────── */
function SessionsTab({ bd }) {
  if (!bd || !bd.session_inventory) return <div style={{ color: '#64748b' }}>Loading...</div>
  const sessions = bd.session_inventory
  return (
    <Card title={`Voice Assessment Sessions (${sessions.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
              {['Patient', 'Instrument', 'Domain', 'Status', 'Items', 'Score', 'Duration', 'Interpretation', 'Date'].map(h => (
                <th key={h} style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#64748b' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sessions.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '7px 10px' }}>{s.patient_name || s.patient_id}</td>
                <td style={{ padding: '7px 10px' }}>{s.instrument}</td>
                <td style={{ padding: '7px 10px', color: '#64748b', fontSize: 12 }}>{s.domain}</td>
                <td style={{ padding: '7px 10px' }}>
                  <span style={{
                    padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                    background: s.status === 'completed' ? '#d1fae5' : s.status === 'abandoned' ? '#fee2e2' : '#dbeafe',
                    color: s.status === 'completed' ? '#065f46' : s.status === 'abandoned' ? '#991b1b' : '#1e40af'
                  }}>{s.status}</span>
                </td>
                <td style={{ padding: '7px 10px' }}>
                  {s.items_completed}/{s.total_items}
                  {s.item_completion_pct != null && <span style={{ color: '#94a3b8', fontSize: 11 }}> ({s.item_completion_pct}%)</span>}
                </td>
                <td style={{ padding: '7px 10px' }}>
                  {s.score != null ? `${s.score}/${s.max_score}` : '-'}
                  {s.score_pct != null && <span style={{ color: '#94a3b8', fontSize: 11 }}> ({s.score_pct}%)</span>}
                </td>
                <td style={{ padding: '7px 10px' }}>{s.duration_seconds ? `${s.duration_seconds}s` : '-'}</td>
                <td style={{ padding: '7px 10px', fontSize: 12 }}>{s.interpretation || '-'}</td>
                <td style={{ padding: '7px 10px', fontSize: 12, color: '#64748b' }}>{(s.started_at || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ── Patients Tab ─────────────────────────────────────────────────── */
function PatientsTab({ bd }) {
  if (!bd || !bd.patient_profiles) return <div style={{ color: '#64748b' }}>Loading...</div>
  const profiles = bd.patient_profiles
  return (
    <Card title={`Voice-Assessed Patients (${profiles.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
              {['Patient', 'Age', 'Gender', 'Disease', 'Voice Sessions', 'Completed', 'Instruments', 'Avg Score', 'Avg Duration'].map(h => (
                <th key={h} style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#64748b' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profiles.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '7px 10px', fontWeight: 600 }}>{p.name || p.patient_id}</td>
                <td style={{ padding: '7px 10px' }}>{p.age || '-'}</td>
                <td style={{ padding: '7px 10px' }}>{p.gender || '-'}</td>
                <td style={{ padding: '7px 10px', fontSize: 12 }}>{p.disease || '-'}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center' }}>{p.total_voice_sessions}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center' }}>{p.completed}</td>
                <td style={{ padding: '7px 10px', fontSize: 12 }}>{p.instruments.join(', ')}</td>
                <td style={{ padding: '7px 10px' }}>{p.mean_score ? `${(p.mean_score * 100).toFixed(1)}%` : '-'}</td>
                <td style={{ padding: '7px 10px' }}>{p.mean_duration ? `${p.mean_duration.toFixed(0)}s` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ── Instruments Tab ──────────────────────────────────────────────── */
function InstrumentsTab({ bd, ov }) {
  if (!bd || !bd.instrument_stats) return <div style={{ color: '#64748b' }}>Loading...</div>
  const stats = bd.instrument_stats
  const comparison = bd.channel_comparison

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Instrument Mapping Stats" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                {['Instrument', 'Domain', 'Total', 'Completed', 'Abandoned', 'Completion %', 'Avg Score', 'Avg Duration'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stats.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '7px 10px', fontWeight: 600 }}>{s.label}</td>
                  <td style={{ padding: '7px 10px', color: '#64748b', fontSize: 12 }}>{s.domain}</td>
                  <td style={{ padding: '7px 10px', textAlign: 'center' }}>{s.total}</td>
                  <td style={{ padding: '7px 10px', textAlign: 'center', color: '#10b981' }}>{s.completed}</td>
                  <td style={{ padding: '7px 10px', textAlign: 'center', color: '#ef4444' }}>{s.abandoned}</td>
                  <td style={{ padding: '7px 10px' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                      background: s.completion_rate >= 0.8 ? '#d1fae5' : s.completion_rate >= 0.5 ? '#fef3c7' : '#fee2e2',
                      color: s.completion_rate >= 0.8 ? '#065f46' : s.completion_rate >= 0.5 ? '#92400e' : '#991b1b'
                    }}>{(s.completion_rate * 100).toFixed(0)}%</span>
                  </td>
                  <td style={{ padding: '7px 10px' }}>{s.mean_score ? `${(s.mean_score * 100).toFixed(1)}%` : '-'}</td>
                  <td style={{ padding: '7px 10px' }}>{s.mean_duration ? `${s.mean_duration.toFixed(0)}s` : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {comparison && (
        <Card title="Channel Comparison: Voice vs Chat" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            {['voice', 'chat'].map(ch => {
              const d = comparison[ch]
              return (
                <div key={ch} style={{ padding: 16, background: '#f8fafc', borderRadius: 10 }}>
                  <h4 style={{ margin: '0 0 12px', color: ch === 'voice' ? '#6366f1' : '#10b981' }}>
                    {ch === 'voice' ? 'Voice AI (STT)' : 'Conversational AI'}
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
                    <div>Total: <b>{d.total}</b></div>
                    <div>Completed: <b>{d.completed}</b></div>
                    <div>Completion: <b>{(d.completion_rate * 100).toFixed(0)}%</b></div>
                    <div>Avg Score: <b>{d.mean_score ? `${(d.mean_score * 100).toFixed(1)}%` : 'N/A'}</b></div>
                    <div>Avg Duration: <b>{d.mean_duration ? `${d.mean_duration.toFixed(0)}s` : 'N/A'}</b></div>
                  </div>
                </div>
              )
            })}
          </div>
        </Card>
      )}
    </div>
  )
}

/* ── Definitions Tab ──────────────────────────────────────────────── */
function DefinitionsTab({ defs }) {
  if (!defs) return <div style={{ color: '#64748b' }}>Loading...</div>
  const sections = [
    { title: 'Mapping Concepts', items: defs.concepts, nameKey: 'name', descKey: 'description' },
    { title: 'Quality Metrics', items: defs.quality_metrics, nameKey: 'name', descKey: 'description' },
    { title: 'Compliance', items: defs.compliance, nameKey: 'ref', descKey: 'note' },
    { title: 'Remediation Strategies', items: defs.remediation, nameKey: 'strategy', descKey: 'description' },
  ]
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {sections.map(sec => (
        <Card key={sec.title} title={sec.title}>
          {(sec.items || []).map((item, i) => (
            <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < sec.items.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{item[sec.nameKey]}</div>
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item[sec.descKey]}</div>
            </div>
          ))}
        </Card>
      ))}

      {defs.supported_instruments && (
        <Card title="Supported Assessment Instruments">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
            {defs.supported_instruments.map((inst, i) => (
              <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8, fontSize: 13 }}>
                <span style={{ fontWeight: 600 }}>{inst.instrument}</span>
                <span style={{ color: '#64748b' }}> — {inst.label}</span>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{inst.domain}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
