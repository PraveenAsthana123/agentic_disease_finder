import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4']

const CONFIDENCE_COLORS = {
  'high': '#10b981',
  'moderate': '#f59e0b',
  'low': '#ef4444',
}

const TYPE_OPTIONS = [
  'general', 'diagnosis', 'differential-diagnosis', 'treatment-decision',
  'medication-adjustment', 'referral', 'risk-assessment', 'discharge-note',
  'prognosis', 'seizure-classification', 'pnes-evaluation',
]

const CONFIDENCE_OPTIONS = ['high', 'moderate', 'low']

export default function ClinicalJudgmentDashboard() {
  const [tab, setTab] = useState('overview')
  const [summary, setSummary] = useState(null)
  const [entries, setEntries] = useState([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [msg, setMsg] = useState('')

  // New entry form
  const [form, setForm] = useState({
    patient_id: '', clinician: '', judgment_type: 'general',
    summary: '', details: '', confidence: 'moderate',
    action_taken: '', follow_up: '',
  })

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [sRes, eRes] = await Promise.all([
        axios.get(`${API}/api/clinical-judgment/summary`),
        axios.get(`${API}/api/clinical-judgment/entries`),
      ])
      setSummary(sRes.data)
      setEntries(eRes.data.entries || [])
    } catch (e) {
      console.error('clinical-judgment load', e)
    }
    setLoading(false)
  }, [])

  useEffect(() => { load() }, [load])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.patient_id.trim() || !form.summary.trim()) {
      setMsg('Patient ID and Summary are required.')
      return
    }
    setSubmitting(true)
    setMsg('')
    try {
      const res = await axios.post(`${API}/api/clinical-judgment/entries`, form)
      if (res.data.ok) {
        setMsg('Clinical judgment recorded successfully.')
        setForm({
          patient_id: '', clinician: '', judgment_type: 'general',
          summary: '', details: '', confidence: 'moderate',
          action_taken: '', follow_up: '',
        })
        load()
      } else {
        setMsg(`Error: ${res.data.error}`)
      }
    } catch (err) {
      setMsg(`Error: ${err.message}`)
    }
    setSubmitting(false)
  }

  const handleDelete = async (id) => {
    try {
      await axios.delete(`${API}/api/clinical-judgment/entries/${id}`)
      load()
    } catch (e) {
      console.error('delete', e)
    }
  }

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'record', label: 'Record Judgment' },
    { id: 'history', label: 'History' },
    { id: 'analytics', label: 'Analytics' },
  ]

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Clinical Judgment Recording</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Psychiatrist/clinician clinical judgment documentation — differential diagnosis, treatment decisions, risk assessments
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 24, borderBottom: '2px solid #e2e8f0' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 20px', border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            color: tab === t.id ? '#3b82f6' : '#64748b',
            background: 'transparent',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            marginBottom: -2,
          }}>{t.label}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#64748b' }}>Loading…</p>}

      {/* ── Overview Tab ──────────────────────────── */}
      {!loading && tab === 'overview' && summary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Summary KPIs" span={2}>
            <div style={{ display: 'flex', gap: 40, justifyContent: 'center', padding: '10px 0' }}>
              <KPI label="Total Judgments" value={summary.total} color="#3b82f6" />
              <KPI label="Judgment Types" value={summary.by_type?.length || 0} color="#8b5cf6" />
              <KPI label="Clinicians" value={summary.by_clinician?.length || 0} color="#10b981" />
            </div>
          </Card>

          <Card title="By Judgment Type">
            {summary.by_type?.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={summary.by_type}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No judgments recorded yet.</p>}
          </Card>

          <Card title="By Confidence Level">
            {summary.by_confidence?.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={summary.by_confidence} dataKey="count" nameKey="confidence"
                    cx="50%" cy="50%" outerRadius={80} label={({ confidence, count }) => `${confidence}: ${count}`}>
                    {summary.by_confidence.map((d, i) => (
                      <Cell key={i} fill={CONFIDENCE_COLORS[d.confidence] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No data yet.</p>}
          </Card>

          <Card title="Recent Judgments" span={2}>
            {summary.recent?.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Clinician</th>
                    <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Summary</th>
                    <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Confidence</th>
                    <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.recent.map(r => (
                    <tr key={r.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 6 }}>{r.patient_id}</td>
                      <td style={{ padding: 6 }}>{r.clinician || '—'}</td>
                      <td style={{ padding: 6 }}><Badge text={r.judgment_type} color="#8b5cf6" /></td>
                      <td style={{ padding: 6, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.summary}</td>
                      <td style={{ padding: 6 }}><Badge text={r.confidence} color={CONFIDENCE_COLORS[r.confidence] || '#64748b'} /></td>
                      <td style={{ padding: 6, color: '#94a3b8' }}>{r.created_at?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No judgments recorded yet. Use the "Record Judgment" tab to add one.</p>}
          </Card>
        </div>
      )}

      {/* ── Record Judgment Tab ───────────────────── */}
      {!loading && tab === 'record' && (
        <Card title="Record New Clinical Judgment" span={2}>
          <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, maxWidth: 700 }}>
            <label style={{ fontSize: 12, color: '#334155' }}>
              Patient ID *
              <input value={form.patient_id} onChange={e => setForm({ ...form, patient_id: e.target.value })}
                style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}
                placeholder="e.g. P001" />
            </label>
            <label style={{ fontSize: 12, color: '#334155' }}>
              Clinician Name
              <input value={form.clinician} onChange={e => setForm({ ...form, clinician: e.target.value })}
                style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}
                placeholder="e.g. Dr. Smith" />
            </label>
            <label style={{ fontSize: 12, color: '#334155' }}>
              Judgment Type
              <select value={form.judgment_type} onChange={e => setForm({ ...form, judgment_type: e.target.value })}
                style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}>
                {TYPE_OPTIONS.map(t => <option key={t} value={t}>{t.replace(/-/g, ' ')}</option>)}
              </select>
            </label>
            <label style={{ fontSize: 12, color: '#334155' }}>
              Confidence Level
              <select value={form.confidence} onChange={e => setForm({ ...form, confidence: e.target.value })}
                style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}>
                {CONFIDENCE_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label style={{ fontSize: 12, color: '#334155', gridColumn: 'span 2' }}>
              Clinical Summary *
              <textarea value={form.summary} onChange={e => setForm({ ...form, summary: e.target.value })}
                rows={2} style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}
                placeholder="Brief clinical judgment summary" />
            </label>
            <label style={{ fontSize: 12, color: '#334155', gridColumn: 'span 2' }}>
              Detailed Notes
              <textarea value={form.details} onChange={e => setForm({ ...form, details: e.target.value })}
                rows={3} style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}
                placeholder="Additional clinical reasoning, observations, differential considerations…" />
            </label>
            <label style={{ fontSize: 12, color: '#334155' }}>
              Action Taken
              <input value={form.action_taken} onChange={e => setForm({ ...form, action_taken: e.target.value })}
                style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}
                placeholder="e.g. Increased LEV dose to 1500mg" />
            </label>
            <label style={{ fontSize: 12, color: '#334155' }}>
              Follow-up Plan
              <input value={form.follow_up} onChange={e => setForm({ ...form, follow_up: e.target.value })}
                style={{ display: 'block', width: '100%', marginTop: 4, padding: '6px 10px', border: '1px solid #cbd5e1', borderRadius: 6, fontSize: 13 }}
                placeholder="e.g. Re-assess in 2 weeks" />
            </label>
            <div style={{ gridColumn: 'span 2', display: 'flex', gap: 12, alignItems: 'center' }}>
              <button type="submit" disabled={submitting} style={{
                padding: '8px 24px', background: '#3b82f6', color: '#fff', border: 'none',
                borderRadius: 6, fontWeight: 600, fontSize: 13, cursor: 'pointer',
                opacity: submitting ? 0.6 : 1,
              }}>{submitting ? 'Saving…' : 'Record Judgment'}</button>
              {msg && <span style={{ fontSize: 12, color: msg.startsWith('Error') ? '#ef4444' : '#10b981' }}>{msg}</span>}
            </div>
          </form>
        </Card>
      )}

      {/* ── History Tab ───────────────────────────── */}
      {!loading && tab === 'history' && (
        <Card title={`All Clinical Judgments (${entries.length})`} span={2}>
          {entries.length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>ID</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Patient</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Clinician</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Type</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Summary</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Confidence</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Action</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Follow-up</th>
                  <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Date</th>
                  <th style={{ padding: 6 }}></th>
                </tr>
              </thead>
              <tbody>
                {entries.map(e => (
                  <tr key={e.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6, color: '#94a3b8' }}>#{e.id}</td>
                    <td style={{ padding: 6, fontWeight: 600 }}>{e.patient_id}</td>
                    <td style={{ padding: 6 }}>{e.clinician || '—'}</td>
                    <td style={{ padding: 6 }}><Badge text={e.judgment_type} color="#8b5cf6" /></td>
                    <td style={{ padding: 6, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.summary}</td>
                    <td style={{ padding: 6 }}><Badge text={e.confidence} color={CONFIDENCE_COLORS[e.confidence] || '#64748b'} /></td>
                    <td style={{ padding: 6, maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.action_taken || '—'}</td>
                    <td style={{ padding: 6, maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.follow_up || '—'}</td>
                    <td style={{ padding: 6, color: '#94a3b8' }}>{e.created_at?.slice(0, 10)}</td>
                    <td style={{ padding: 6 }}>
                      <button onClick={() => handleDelete(e.id)} title="Delete"
                        style={{ border: 'none', background: 'none', color: '#ef4444', cursor: 'pointer', fontSize: 14 }}>✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No clinical judgments recorded yet.</p>}
        </Card>
      )}

      {/* ── Analytics Tab ─────────────────────────── */}
      {!loading && tab === 'analytics' && summary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          <Card title="Judgments by Clinician">
            {summary.by_clinician?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={summary.by_clinician} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis type="category" dataKey="clinician" width={100} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No data yet.</p>}
          </Card>

          <Card title="Confidence Distribution">
            {summary.by_confidence?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={summary.by_confidence} dataKey="count" nameKey="confidence"
                    cx="50%" cy="50%" innerRadius={50} outerRadius={90}
                    label={({ confidence, count }) => `${confidence}: ${count}`}>
                    {summary.by_confidence.map((d, i) => (
                      <Cell key={i} fill={CONFIDENCE_COLORS[d.confidence] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No data yet.</p>}
          </Card>

          <Card title="Judgment Types Breakdown" span={2}>
            {summary.by_type?.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={summary.by_type}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={70} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>Record judgments to see analytics.</p>}
          </Card>
        </div>
      )}
    </div>
  )
}
