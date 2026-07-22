import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#94a3b8' }

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const bg = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

function ModelTag({ model }) {
  return (
    <span style={{
      background: '#eff6ff', color: '#3b82f6', border: '1px solid #bfdbfe',
      borderRadius: 4, padding: '2px 6px', fontSize: 10, fontWeight: 500, marginRight: 4, marginBottom: 2, display: 'inline-block'
    }}>
      {model}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function NeuroAdvancementsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/neuro-advancements/overview`),
      axios.get(`${API_URL}/api/neuro-advancements/breakdown`),
      axios.get(`${API_URL}/api/neuro-advancements/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Neuro Advancements...</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#ef4444' }}>neuro_advancements.json not found</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'modalities', label: 'By Modality' },
    { id: 'models', label: 'AI Models' },
    { id: 'crossmodal', label: 'Cross-Modal' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const s = overview.summary || {}

  return (
    <div style={{ padding: '16px 20px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>
          {overview.title || 'Neuro Advancements'}
        </h2>
        <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
          Updated: {overview.updated_at || 'N/A'} &middot; {s.total_modalities || 0} modalities &middot; {s.unique_ai_models || 0} AI models
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
            border: tab === t.id ? '1.5px solid #3b82f6' : '1px solid #e2e8f0',
            background: tab === t.id ? '#eff6ff' : '#fff',
            color: tab === t.id ? '#3b82f6' : '#64748b',
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Modalities" value={s.total_modalities || 0} />
              <KPI label="Built" value={s.built || 0} sub="pipelines active" />
              <KPI label="AI Models" value={s.unique_ai_models || 0} sub="unique across modalities" />
              <KPI label="Biomarkers" value={s.total_biomarkers || 0} sub="tracked markers" />
              <KPI label="Cross-Modal" value={s.cross_modal_count || 0} sub="fusion ideas" />
            </div>
          </Card>

          <Card title="Status Distribution">
            {(overview.status_distribution || []).length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.status_distribution || []).map((_, i) => (
                      <Cell key={i} fill={[STATUS_COLORS.built, STATUS_COLORS.partial, STATUS_COLORS.planned][i] || COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 12 }}>No data</div>}
          </Card>

          <Card title="AI Models per Modality">
            {(overview.models_per_modality || []).length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={overview.models_per_modality}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 12 }}>No data</div>}
          </Card>

          <Card title="All Modalities" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Code</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Advancement</th>
                    <th style={thStyle}>AI Models</th>
                    <th style={thStyle}>Biomarker</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.modalities_table || []).map((m, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, fontFamily: 'monospace' }}>{m.code}</td>
                      <td style={tdStyle}>{m.name}</td>
                      <td style={{ ...tdStyle, maxWidth: 250, fontSize: 11 }}>{m.advancement}</td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{m.ai_models}</td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{m.biomarker}</td>
                      <td style={tdStyle}><StatusBadge status={m.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'modalities' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          {(breakdown.per_modality || []).map((m, i) => (
            <Card key={i} title={`${m.code} — ${m.name}`}>
              <div style={{ marginBottom: 8 }}>
                <StatusBadge status={m.status} />
              </div>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 8 }}>{m.advancement}</div>
              <div style={{ marginBottom: 6 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: '#64748b' }}>AI Models: </span>
                {(m.ai_models || []).map((model, j) => <ModelTag key={j} model={model} />)}
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                <strong>Biomarker:</strong> {m.biomarker || 'N/A'}
              </div>
              {m.note && (
                <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 6, fontStyle: 'italic' }}>{m.note}</div>
              )}
            </Card>
          ))}
        </div>
      )}

      {tab === 'models' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="AI Model Distribution" span={2}>
            {(overview.model_distribution || []).length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={overview.model_distribution} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 12 }}>No data</div>}
          </Card>

          <Card title="AI Model Index" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Model</th>
                    <th style={thStyle}>Used By</th>
                    <th style={thStyle}>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.ai_model_index || []).map((m, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{m.model}</td>
                      <td style={tdStyle}>
                        {(m.used_by || []).map((code, j) => (
                          <span key={j} style={{
                            background: '#f0fdf4', color: '#16a34a', border: '1px solid #bbf7d0',
                            borderRadius: 4, padding: '1px 6px', fontSize: 10, fontWeight: 500, marginRight: 3
                          }}>{code}</span>
                        ))}
                      </td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{m.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Biomarker Index" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Modality</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Biomarker</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.biomarker_index || []).map((b, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, fontFamily: 'monospace' }}>{b.modality}</td>
                      <td style={tdStyle}>{b.name}</td>
                      <td style={tdStyle}>{b.biomarker}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'crossmodal' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16, maxWidth: 800 }}>
          <Card title="Cross-Modal Advancement Opportunities">
            <div style={{ fontSize: 12, color: '#475569', marginBottom: 12 }}>
              These represent the frontier of multimodal neurophysiology AI — combining signals across modalities
              for more accurate, interpretable, and robust clinical decision support.
            </div>
            {(breakdown.cross_modal_advancements || []).map((adv, i) => (
              <div key={i} style={{
                padding: '10px 14px', marginBottom: 8, borderRadius: 6,
                background: i % 2 === 0 ? '#f0f9ff' : '#faf5ff',
                border: `1px solid ${i % 2 === 0 ? '#bae6fd' : '#e9d5ff'}`,
                fontSize: 13, color: '#1e293b'
              }}>
                <span style={{
                  display: 'inline-block', width: 22, height: 22, borderRadius: '50%',
                  background: COLORS[i % COLORS.length], color: '#fff', textAlign: 'center',
                  lineHeight: '22px', fontSize: 11, fontWeight: 700, marginRight: 10
                }}>{i + 1}</span>
                {adv}
              </div>
            ))}
          </Card>

          <Card title="Modality Categories">
            {defs && (defs.modality_categories || []).map((cat, i) => (
              <div key={i} style={{
                padding: '10px 14px', marginBottom: 8, borderRadius: 6,
                background: '#f8fafc', border: '1px solid #e2e8f0'
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{cat.category}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{cat.description}</div>
                <div>
                  {(cat.modalities || []).map((m, j) => (
                    <span key={j} style={{
                      background: '#eff6ff', color: '#3b82f6', border: '1px solid #bfdbfe',
                      borderRadius: 4, padding: '2px 6px', fontSize: 10, fontWeight: 500, marginRight: 4
                    }}>{m}</span>
                  ))}
                </div>
              </div>
            ))}
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          <Card title="Status Legend">
            {(defs.status_legend || []).map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <span style={{
                  display: 'inline-block', width: 12, height: 12, borderRadius: 3,
                  background: s.color, flexShrink: 0
                }} />
                <span style={{ fontSize: 12, color: '#334155' }}>{s.description}</span>
              </div>
            ))}
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Term</th>
                    <th style={thStyle}>Definition</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Clinical Notes">
            {(defs.clinical_notes || []).map((n, i) => (
              <div key={i} style={{
                padding: '8px 12px', marginBottom: 6, borderRadius: 4,
                background: '#fefce8', border: '1px solid #fde68a', fontSize: 12, color: '#92400e'
              }}>
                {n}
              </div>
            ))}
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 11, color: '#475569', marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </div>
      )}
    </div>
  )
}
