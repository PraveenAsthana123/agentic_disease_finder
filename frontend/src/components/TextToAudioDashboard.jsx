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

const TIER_COLORS = {
  short: '#10b981',
  medium: '#3b82f6',
  long: '#f59e0b',
  extended: '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'text-inventory', label: 'Text Inventory' },
  { id: 'report-inventory', label: 'Report Inventory' },
  { id: 'patient-profiles', label: 'Patient Profiles' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function TextToAudioDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/text-to-audio/overview`),
      axios.get(`${API_URL}/api/text-to-audio/breakdown`),
      axios.get(`${API_URL}/api/text-to-audio/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Text-to-Audio AI data...</div>
  if (err) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {err}</div>
  if (!ov?.available) return <div style={{ padding: 40, color: '#64748b' }}>{ov?.message || 'No data available.'}</div>

  const kpis = ov.kpis || []

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Text-to-Audio AI</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Clinical TTS synthesis pipeline — text corpus, audio generation estimates, patient audio profiles
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(kpis, ov)}
      {tab === 'text-inventory' && renderTextInventory(bd)}
      {tab === 'report-inventory' && renderReportInventory(bd)}
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
        {/* Text length distribution */}
        <Card title="Text Length Distribution">
          {ov.text_length_distribution?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.text_length_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="tier" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Texts">
                  {ov.text_length_distribution.map((d, i) => (
                    <Cell key={i} fill={TIER_COLORS[d.tier] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        {/* Role distribution pie */}
        <Card title="Source Role Distribution">
          {ov.role_distribution?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.role_distribution} dataKey="count" nameKey="role" cx="50%" cy="50%"
                  outerRadius={80} label={({ role, count }) => `${role}: ${count}`}>
                  {ov.role_distribution.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>
      </div>

      {/* Daily activity */}
      <Card title="Daily Synthesis Activity">
        {ov.daily_activity?.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={ov.daily_activity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>
    </>
  )
}

function renderTextInventory(bd) {
  const items = bd?.text_inventory || []
  return (
    <Card title={`Synthesizable Text Corpus (${items.length})`} span={2}>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
              {['ID', 'Role', 'Preview', 'Words', 'Chars', 'Tier', 'Est. Duration', 'Timestamp'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((t, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}>{t.id}</td>
                <td><Badge text={t.role} color={t.role === 'assistant' ? '#3b82f6' : '#10b981'} /></td>
                <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.text_preview}</td>
                <td>{t.word_count}</td>
                <td>{t.char_count}</td>
                <td><Badge text={t.length_tier} color={TIER_COLORS[t.length_tier] || '#6b7280'} /></td>
                <td>{t.est_duration_sec}s</td>
                <td style={{ fontSize: 11, color: '#94a3b8' }}>{(t.ts_utc || '').slice(0, 16)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderReportInventory(bd) {
  const items = bd?.report_inventory || []
  return (
    <Card title={`Clinical Report Audio Queue (${items.length})`} span={2}>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
              {['ID', 'Patient', 'Instrument', 'Interpretation', 'Chars', 'Est. Duration', 'Level', 'Date'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}>{r.id}</td>
                <td>{r.patient_id}</td>
                <td><Badge text={r.instrument} color="#8b5cf6" /></td>
                <td style={{ maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.interpretation}</td>
                <td>{r.char_count}</td>
                <td>{r.est_duration_sec}s</td>
                <td>{r.level && <Badge text={r.level} color={TIER_COLORS[r.level] || '#6b7280'} />}</td>
                <td style={{ fontSize: 11, color: '#94a3b8' }}>{(r.created_at || '').slice(0, 10)}</td>
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
    <Card title={`Patient Audio Profiles (${profiles.length})`} span={2}>
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
              <span><strong>{p.n_reports}</strong> reports</span>
              <span><strong>{p.total_report_chars}</strong> chars</span>
              <span><strong>{p.est_audio_sec}s</strong> est. audio</span>
            </div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {(p.instruments || []).map((inst, j) => (
                <Badge key={j} text={inst} color="#8b5cf6" />
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
              <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
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
      <Card title="TTS & Audio Synthesis Concepts">
        <div style={{ display: 'grid', gap: 10 }}>
          {(defs.concepts || []).map((c, i) => (
            <div key={i} style={{ borderLeft: '3px solid #3b82f6', paddingLeft: 12 }}>
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

      {/* Audio categories */}
      <Card title="Audio Output Categories">
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {(defs.audio_categories || []).map((c, i) => (
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
