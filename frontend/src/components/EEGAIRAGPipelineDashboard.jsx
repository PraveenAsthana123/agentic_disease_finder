import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316', '#14b8a6', '#a855f7']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function StatusBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'built' ? '#10b981' : s === 'partial' ? '#f59e0b' : s === 'planned' ? '#ef4444' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(status || 'N/A')}</span>
  )
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const STATUS_COLORS = { built: '#10b981', partial: '#f59e0b', planned: '#ef4444' }

export default function EEGAIRAGPipelineDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/eeg-ai-rag-pipeline/overview`),
      axios.get(`${API_URL}/api/eeg-ai-rag-pipeline/breakdown`),
      axios.get(`${API_URL}/api/eeg-ai-rag-pipeline/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG AI RAG Pipeline...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Pipeline data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'steps', label: 'Pipeline Steps' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = ov.kpis || {}
  const statusDist = (ov.status_distribution || []).map(d => ({ ...d, fill: STATUS_COLORS[d.status] || '#64748b' }))
  const phaseData = (ov.phases || []).map((p, i) => ({ ...p, fill: COLORS[i % COLORS.length] }))

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: 600, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card title="EEG → AI → RAG Pipeline KPIs" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Total Steps" value={kpis.total_steps} color="#1e293b" />
              <KPI label="Built" value={kpis.built} color="#10b981" />
              <KPI label="Partial" value={kpis.partial} color="#f59e0b" />
              <KPI label="Planned" value={kpis.planned} color="#ef4444" />
              <KPI label="Completion %" value={kpis.completion_pct} sub="%" color="#8b5cf6" />
              <KPI label="Phases" value={kpis.phases} color="#3b82f6" />
            </div>
          </Card>

          {/* Status Distribution Pie */}
          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={statusDist} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={90} label={({ status, count }) => `${status}: ${count}`}>
                  {statusDist.map((d, i) => (
                    <Cell key={i} fill={d.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Phase Completion Bar */}
          <Card title="Phase Completion (%)" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={phaseData} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis type="category" dataKey="phase" tick={{ fontSize: 12 }} width={110} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="completion_pct" radius={[0, 4, 4, 0]} name="Completion %">
                  {phaseData.map((d, i) => (
                    <Cell key={i} fill={d.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pipeline Flow Table */}
          <Card title={`Pipeline Flow (${(ov.steps || []).length} steps)`} span={3}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['#', 'Step', 'Detail', 'Status', 'Where'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.steps || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.n}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.step}</td>
                      <td style={{ padding: '6px 10px', color: '#475569', maxWidth: 400 }}>{s.detail}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={s.status} /></td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12, color: '#64748b' }}>{s.where || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Honest Note */}
          {ov.honest_note && (
            <Card title="Honest Note" span={3}>
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{ov.honest_note}</p>
            </Card>
          )}
        </div>
      )}

      {tab === 'steps' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Phase Summary Cards */}
          <Card title="Phase Summary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              {(ov.phases || []).map((p, i) => (
                <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 16 }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 8 }}>{p.phase}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6, fontSize: 13 }}>
                    <span style={{ color: '#64748b' }}>Total:</span><span style={{ fontWeight: 600 }}>{p.total}</span>
                    <span style={{ color: '#64748b' }}>Built:</span><span style={{ fontWeight: 600, color: '#10b981' }}>{p.built}</span>
                    <span style={{ color: '#64748b' }}>Partial:</span><span style={{ fontWeight: 600, color: '#f59e0b' }}>{p.partial}</span>
                    <span style={{ color: '#64748b' }}>Planned:</span><span style={{ fontWeight: 600, color: '#ef4444' }}>{p.planned}</span>
                    <span style={{ color: '#64748b' }}>Completion:</span><span style={{ fontWeight: 600, color: '#8b5cf6' }}>{p.completion_pct}%</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Full Steps Table */}
          <Card title={`All Pipeline Steps (${bd.total})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Step #', 'Phase', 'Step Name', 'Detail', 'Status', 'Where'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.steps || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.n}</td>
                      <td style={{ padding: '6px 10px', color: '#3b82f6', fontWeight: 600, whiteSpace: 'nowrap' }}>{s.phase || '--'}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.step}</td>
                      <td style={{ padding: '6px 10px', color: '#475569', maxWidth: 350 }}>{s.detail}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={s.status} /></td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12, color: '#64748b' }}>{s.where || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Phase Descriptions */}
          <Card title="Pipeline Phases">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Phase', 'Steps', 'Description'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.phases || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: '#1e293b', whiteSpace: 'nowrap' }}>{p.name}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: '#3b82f6', whiteSpace: 'nowrap' }}>{p.steps}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', lineHeight: 1.5 }}>{p.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Glossary */}
          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Term', 'Definition'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: '#1e293b', whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', lineHeight: 1.5 }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Clinical Notes */}
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 13, color: '#475569', lineHeight: 1.7, marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          {/* References */}
          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6, marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </div>
      )}
    </div>
  )
}
