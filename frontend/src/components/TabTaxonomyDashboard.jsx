import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#4caf50', '#7c4dff', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { built: '#4caf50', partial: '#ff9800', planned: '#f44336' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Badge({ status }) {
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

const cardStyle = {
  background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
  padding: 20, marginBottom: 18
}

function kpiStyle(color) {
  return {
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 130
  }
}

export default function TabTaxonomyDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [catFilter, setCatFilter] = useState('all')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/tab-taxonomy/overview`),
          axios.get(`${API_URL}/api/tab-taxonomy/breakdown`),
          axios.get(`${API_URL}/api/tab-taxonomy/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Tab Taxonomy data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading tab taxonomy...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fed7aa', borderRadius: 8, color: '#92400e' }}>
      Tab taxonomy data not available
    </div>
  )

  const kpis = overview.kpis || {}
  const tabs = ['overview', 'patient-master', 'role-ops', 'ai-caps', 'transformation', 'definitions']

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ marginBottom: 4 }}>{overview.title || 'Tab Taxonomy'}</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        {overview.note} {overview.updated_at && <span>| Updated: {overview.updated_at}</span>}
      </p>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#1e88e5' : '#f1f5f9',
            color: tab === t ? '#fff' : '#475569', fontWeight: tab === t ? 600 : 400, fontSize: 13
          }}>
            {t === 'overview' ? 'Overview' : t === 'patient-master' ? 'Patient Master' :
             t === 'role-ops' ? 'Role Operations' : t === 'ai-caps' ? 'AI Capabilities' :
             t === 'transformation' ? 'As-Is / To-Be' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ──────────────────────────────────────── */}
      {tab === 'overview' && (
        <div>
          {/* KPIs */}
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
            <div style={kpiStyle('#1e88e5')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#1e88e5' }}>{fmt(kpis.total_tabs)}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Total Tabs</div>
            </div>
            <div style={kpiStyle('#7c4dff')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#7c4dff' }}>{fmt(kpis.categories)}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Categories</div>
            </div>
            <div style={kpiStyle('#4caf50')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#4caf50' }}>{fmt(kpis.built)}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Built</div>
            </div>
            <div style={kpiStyle('#ff9800')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#ff9800' }}>{fmt(kpis.built_pct)}%</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Built %</div>
            </div>
            <div style={kpiStyle('#00bcd4')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#00bcd4' }}>{fmt(kpis.mapped)}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Mapped</div>
            </div>
            <div style={kpiStyle('#e91e63')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#e91e63' }}>{fmt(kpis.patient_master_count)}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Patient Master</div>
            </div>
            <div style={kpiStyle('#607d8b')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#607d8b' }}>{fmt(kpis.role_ops_count)}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Role Ops</div>
            </div>
            <div style={kpiStyle('#9c27b0')}>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#9c27b0' }}>{fmt(kpis.ai_caps_count)}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>AI Caps</div>
            </div>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 18, marginBottom: 22 }}>
            {/* Tabs per category */}
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px' }}>Tabs per Category</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.tabs_per_category}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#1e88e5" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Status distribution pie */}
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px' }}>Status Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.status_distribution || []).map((_, i) => (
                      <Cell key={i} fill={[STATUS_COLORS.built, STATUS_COLORS.partial, STATUS_COLORS.planned][i] || COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Mapping coverage pie */}
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px' }}>Mapping Coverage</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.mapping_coverage} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.mapping_coverage || []).map((_, i) => (
                      <Cell key={i} fill={['#4caf50', '#e0e0e0'][i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* All tabs summary table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px' }}>All Tabs Summary</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Category</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Tab</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Mapped</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.tabs_summary || []).map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 12 }}>{t.category}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 500 }}>{t.label}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}><Badge status={t.status} /></td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{t.has_mapping ? '\u2705' : '\u2014'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Patient Master Tab ────────────────────────────────── */}
      {tab === 'patient-master' && breakdown?.categories && (
        <div>
          {(() => {
            const cat = breakdown.categories.find(c => c.key === 'patient_master_tabs')
            if (!cat) return <p>No data</p>
            return (
              <>
                <div style={{ ...cardStyle, background: '#e3f2fd' }}>
                  <strong>Patient Master — Self-Service Portal</strong>
                  <span style={{ marginLeft: 12, fontSize: 13, color: '#1565c0' }}>
                    {cat.built}/{cat.total} tabs built
                  </span>
                </div>
                {cat.tabs.map((t, i) => (
                  <div key={i} style={cardStyle}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <strong>{t.label}</strong>
                      <Badge status={t.status} />
                    </div>
                    {t.description && (
                      <p style={{ color: '#475569', fontSize: 13, margin: '4px 0' }}>
                        <em>Captures:</em> {t.description}
                      </p>
                    )}
                    {t.maps_to && (
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 6 }}>
                        <code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 3 }}>{t.maps_to}</code>
                      </div>
                    )}
                  </div>
                ))}
              </>
            )
          })()}
        </div>
      )}

      {/* ── Role Operations Tab ───────────────────────────────── */}
      {tab === 'role-ops' && breakdown?.categories && (
        <div>
          {(() => {
            const cat = breakdown.categories.find(c => c.key === 'role_operational_tabs')
            if (!cat) return <p>No data</p>
            return (
              <>
                <div style={{ ...cardStyle, background: '#e8f5e9' }}>
                  <strong>Role Operational Tabs</strong>
                  <span style={{ marginLeft: 12, fontSize: 13, color: '#2e7d32' }}>
                    {cat.built}/{cat.total} tabs built
                  </span>
                </div>
                {cat.tabs.map((t, i) => (
                  <div key={i} style={cardStyle}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <strong>{t.label}</strong>
                      <Badge status={t.status} />
                    </div>
                    {t.metric && (
                      <p style={{ color: '#475569', fontSize: 13, margin: '4px 0' }}>
                        <em>Metric:</em> {t.metric}
                      </p>
                    )}
                    {t.maps_to && (
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 6 }}>
                        <code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 3 }}>{t.maps_to}</code>
                      </div>
                    )}
                  </div>
                ))}
              </>
            )
          })()}
        </div>
      )}

      {/* ── AI Capabilities Tab ───────────────────────────────── */}
      {tab === 'ai-caps' && breakdown?.categories && (
        <div>
          {(() => {
            const cat = breakdown.categories.find(c => c.key === 'ai_capability_tabs')
            if (!cat) return <p>No data</p>
            return (
              <>
                <div style={{ ...cardStyle, background: '#f3e5f5' }}>
                  <strong>AI Capability Tabs</strong>
                  <span style={{ marginLeft: 12, fontSize: 13, color: '#7b1fa2' }}>
                    {cat.built}/{cat.total} tabs built
                  </span>
                </div>
                {cat.tabs.map((t, i) => (
                  <div key={i} style={cardStyle}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <strong>{t.label}</strong>
                      <Badge status={t.status} />
                    </div>
                    {t.maps_to && (
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 6 }}>
                        <code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 3 }}>{t.maps_to}</code>
                      </div>
                    )}
                  </div>
                ))}
              </>
            )
          })()}
        </div>
      )}

      {/* ── As-Is / To-Be Transformation Tab ──────────────────── */}
      {tab === 'transformation' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
            <div style={{ ...cardStyle, borderLeft: '4px solid #f44336' }}>
              <h4 style={{ margin: '0 0 10px', color: '#c62828' }}>As-Is (Current State)</h4>
              <p style={{ color: '#475569', fontSize: 14, lineHeight: 1.6 }}>
                {overview.as_is_to_be?.as_is || 'Not available'}
              </p>
            </div>
            <div style={{ ...cardStyle, borderLeft: '4px solid #4caf50' }}>
              <h4 style={{ margin: '0 0 10px', color: '#2e7d32' }}>To-Be (Target State)</h4>
              <p style={{ color: '#475569', fontSize: 14, lineHeight: 1.6 }}>
                {overview.as_is_to_be?.to_be || 'Not available'}
              </p>
            </div>
          </div>
          <div style={{ ...cardStyle, marginTop: 18 }}>
            <h4 style={{ margin: '0 0 12px' }}>Transformation Progress</h4>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ flex: 1, background: '#e0e0e0', borderRadius: 6, height: 20, overflow: 'hidden' }}>
                <div style={{
                  width: `${kpis.built_pct || 0}%`, height: '100%',
                  background: 'linear-gradient(90deg, #4caf50, #81c784)', borderRadius: 6,
                  transition: 'width 0.5s'
                }} />
              </div>
              <span style={{ fontWeight: 600, color: '#4caf50', minWidth: 50 }}>{kpis.built_pct || 0}%</span>
            </div>
            <p style={{ color: '#64748b', fontSize: 12, marginTop: 8 }}>
              {kpis.built} of {kpis.total_tabs} tabs built across {kpis.categories} categories
            </p>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────── */}
      {tab === 'definitions' && defs?.available && (
        <div>
          {/* Status legend */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px' }}>Status Legend</h4>
            <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
              {(defs.status_legend || []).map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    width: 14, height: 14, borderRadius: 3, background: s.color, display: 'inline-block'
                  }} />
                  <span style={{ fontSize: 13 }}><strong>{s.status}</strong>: {s.meaning}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Glossary */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px' }}>Glossary ({(defs.glossary || []).length} terms)</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>{g.term}</strong>
                  <p style={{ color: '#475569', fontSize: 12, margin: '4px 0 0' }}>{g.definition}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Clinical notes */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px' }}>Clinical Notes</h4>
            <ul style={{ margin: 0, paddingLeft: 20, color: '#475569', fontSize: 13 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </div>

          {/* References */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px' }}>References</h4>
            {(defs.references || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 8, fontSize: 13 }}>
                <strong>{r.label}</strong>
                <span style={{ color: '#64748b', marginLeft: 8 }}>{r.note}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
