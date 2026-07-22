import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#8b5cf6', '#22c55e', '#f97316', '#ef4444', '#14b8a6', '#ec4899', '#eab308']
const TYPE_COLORS = { Intake: '#3b82f6', 'Auto-extracted': '#22c55e', Deferred: '#8b5cf6' }

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

function Badge({ label, color }) {
  return (
    <span style={{
      background: `${color}22`, color: color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {label}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function OnboardingIntakeDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/onboarding-intake/overview`),
      axios.get(`${API_URL}/api/onboarding-intake/breakdown`),
      axios.get(`${API_URL}/api/onboarding-intake/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Onboarding Intake...</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#ef4444' }}>onboarding_intake.json not found</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'intake-groups', label: 'Intake Groups' },
    { id: 'auto-extract', label: 'Auto-Extract' },
    { id: 'deferred', label: 'Deferred' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const s = overview.summary || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Patient Onboarding Intake Dashboard
      </h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{overview.goal}</p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontSize: 12, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'space-around' }}>
              <KPI label="Intake Fields" value={s.true_intake_fields} sub="captured at registration" />
              <KPI label="Deferred Fields" value={s.deferred_fields} sub="captured over time" />
              <KPI label="Reduction" value={s.reduction} sub="field reduction" />
              <KPI label="Time Saved" value={s.time_saved} sub="active intake time" />
              <KPI label="Groups" value={s.total_groups} sub="intake field groups" />
              <KPI label="Extract Sources" value={s.extraction_sources} sub="document types" />
              <KPI label="Deferred Sections" value={s.deferred_sections_count} sub="longitudinal capture" />
            </div>
          </Card>

          <Card title="Intake vs Deferred Fields">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.intake_vs_deferred} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.intake_vs_deferred || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Fields per Intake Group">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.group_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Onboarding Steps" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Step</th>
                  <th style={thStyle}>Title</th>
                  <th style={thStyle}>Approach</th>
                </tr>
              </thead>
              <tbody>
                {(overview.steps_table || []).map((row, i) => (
                  <tr key={i}>
                    <td style={tdStyle}><Badge label={`Step ${row.step}`} color={COLORS[i % COLORS.length]} /></td>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{row.title}</td>
                    <td style={tdStyle}>{row.approach}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* Intake Groups Tab */}
      {tab === 'intake-groups' && breakdown?.step1 && (
        <div>
          <Card title={`Step 1: ${breakdown.step1.title}`}>
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
              Approach: <strong>{breakdown.step1.approach}</strong> — {breakdown.step1.total_intake_fields} total fields
            </p>
          </Card>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            {(breakdown.step1.groups || []).map((g, i) => (
              <Card key={i} title={g.group}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <Badge label={`${g.n} fields`} color="#3b82f6" />
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {(g.fields || []).map((f, j) => (
                    <span key={j} style={{
                      background: '#f1f5f9', borderRadius: 4, padding: '2px 8px',
                      fontSize: 11, color: '#334155'
                    }}>{f}</span>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Auto-Extract Tab */}
      {tab === 'auto-extract' && breakdown?.step2 && (
        <div>
          <Card title={`Step 2: ${breakdown.step2.title}`}>
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
              Approach: <strong>{breakdown.step2.approach}</strong>
            </p>
            {breakdown.step2.note && (
              <p style={{ fontSize: 11, color: '#22c55e', fontStyle: 'italic' }}>{breakdown.step2.note}</p>
            )}
          </Card>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            {(breakdown.step2.extracts || []).map((e, i) => (
              <Card key={i} title={e.doc}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <Badge label={`${e.fills.length} fields auto-filled`} color="#22c55e" />
                </div>
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Auto-extracted fields:</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {e.fills.map((f, j) => (
                      <span key={j} style={{
                        background: '#22c55e22', borderRadius: 4, padding: '2px 8px',
                        fontSize: 11, color: '#16a34a', border: '1px solid #22c55e55'
                      }}>{f}</span>
                    ))}
                  </div>
                </div>
                {/* Flow arrow */}
                <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: '#64748b' }}>
                  <span style={{ background: '#f1f5f9', borderRadius: 4, padding: '2px 8px' }}>Upload</span>
                  <span>→</span>
                  <span style={{ background: '#f1f5f9', borderRadius: 4, padding: '2px 8px' }}>AI Parse</span>
                  <span>→</span>
                  <span style={{ background: '#22c55e22', borderRadius: 4, padding: '2px 8px', color: '#16a34a' }}>Auto-fill</span>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Deferred Tab */}
      {tab === 'deferred' && breakdown?.step3 && (
        <div>
          <Card title={`Step 3: ${breakdown.step3.title}`}>
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
              Approach: <strong>{breakdown.step3.approach}</strong> — Est. {breakdown.step3.deferred_field_estimate} fields
            </p>
            {breakdown.step3.note && (
              <p style={{ fontSize: 11, color: '#8b5cf6', fontStyle: 'italic' }}>{breakdown.step3.note}</p>
            )}
          </Card>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            {(breakdown.step3.deferred_sections || []).map((d, i) => (
              <Card key={i} title={d.section}>
                <Badge label="Deferred" color="#8b5cf6" />
                <p style={{ fontSize: 12, color: '#334155', marginTop: 8 }}>{d.capture}</p>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs?.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Step Descriptions" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Step</th>
                  <th style={thStyle}>Title</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.step_descriptions || []).map((s, i) => (
                  <tr key={i}>
                    <td style={tdStyle}><Badge label={`Step ${s.step}`} color={COLORS[i % COLORS.length]} /></td>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{s.title}</td>
                    <td style={{ ...tdStyle, fontSize: 11 }}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Field Classification Legend" span={2}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
              {(defs.field_classification_legend || []).map((l, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, minWidth: 260 }}>
                  <Badge label={l.type} color={l.color} />
                  <span style={{ fontSize: 11, color: '#64748b' }}>{l.description}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Term</th>
                  <th style={thStyle}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={{ ...tdStyle, fontSize: 11 }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 11, color: '#334155', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 16 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </div>
      )}
    </div>
  )
}
