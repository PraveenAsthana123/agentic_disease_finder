import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const LEVEL_COLORS = { normal: '#22c55e', mild: '#eab308', moderate: '#f97316', severe: '#ef4444', critical: '#991b1b' }

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

function LevelBadge({ level }) {
  const bg = LEVEL_COLORS[level] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'capitalize'
    }}>
      {level}
    </span>
  )
}

function DirectionBadge({ direction }) {
  const isHigher = direction === 'higher_better'
  const bg = isHigher ? '#3b82f6' : '#f97316'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {isHigher ? '↑ Higher Better' : '↓ Lower Better'}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function AssessmentInstrumentsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [roleFilter, setRoleFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/assessment-instruments/overview`),
      axios.get(`${API_URL}/api/assessment-instruments/breakdown`),
      axios.get(`${API_URL}/api/assessment-instruments/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Assessment Instruments...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'all-instruments', 'by-role', 'bands', 'definitions']
  const k = overview.kpis || {}
  const instruments = overview.instruments || []
  const allBd = (breakdown && breakdown.instruments) || []
  const roles = [...new Set(instruments.map(i => i.role))].sort()

  /* ── Overview Tab ── */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, justifyContent: 'center' }}>
          <KPI label="Instruments" value={k.total_instruments} />
          <KPI label="Roles" value={k.unique_roles} />
          <KPI label="Interpretation Bands" value={k.total_bands} />
          <KPI label="Sub-Domains" value={k.total_domains} />
          <KPI label="Scoring Types" value={k.scoring_types} />
          <KPI label="Avg Max Score" value={k.avg_max_score} />
        </div>
      </Card>

      <Card title="Instruments by Role">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.role_distribution} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis type="category" dataKey="name" width={95} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Scoring Method">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.scoring_distribution} dataKey="value" nameKey="name"
              cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
              {(overview.scoring_distribution || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Score Direction">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.direction_distribution} dataKey="value" nameKey="name"
              cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
              {(overview.direction_distribution || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Max Score per Instrument" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.max_score_chart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="All Instruments Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>#</th>
                <th style={thStyle}>Instrument</th>
                <th style={thStyle}>Role</th>
                <th style={thStyle}>Max</th>
                <th style={thStyle}>Scoring</th>
                <th style={thStyle}>Direction</th>
                <th style={thStyle}>Bands</th>
                <th style={thStyle}>Domains</th>
              </tr>
            </thead>
            <tbody>
              {instruments.map((inst, i) => (
                <tr key={inst.id}>
                  <td style={tdStyle}>{i + 1}</td>
                  <td style={tdStyle}><strong>{inst.icon} {inst.id}</strong><br /><span style={{ fontSize: 10, color: '#94a3b8' }}>{inst.name}</span></td>
                  <td style={tdStyle}>{inst.role}</td>
                  <td style={tdStyle}>{inst.max}</td>
                  <td style={tdStyle}>{inst.scoring}</td>
                  <td style={tdStyle}><DirectionBadge direction={inst.direction} /></td>
                  <td style={tdStyle}>{inst.band_count}</td>
                  <td style={tdStyle}>{inst.domain_count || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  /* ── All Instruments Tab ── */
  const renderAllInstruments = () => {
    const filtered = roleFilter === 'all' ? allBd : allBd.filter(i => i.role === roleFilter)
    return (
      <div>
        <div style={{ marginBottom: 16 }}>
          <select value={roleFilter} onChange={e => setRoleFilter(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid #cbd5e1', fontSize: 13 }}>
            <option value="all">All Roles ({allBd.length})</option>
            {roles.map(r => (
              <option key={r} value={r}>{r} ({allBd.filter(i => i.role === r).length})</option>
            ))}
          </select>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          {filtered.map(inst => (
            <Card key={inst.id} title={`${inst.icon} ${inst.id} — ${inst.name}`}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                <span style={{ fontSize: 11, color: '#64748b', background: '#f1f5f9', padding: '2px 8px', borderRadius: 4 }}>{inst.role}</span>
                <span style={{ fontSize: 11, color: '#64748b', background: '#f1f5f9', padding: '2px 8px', borderRadius: 4 }}>Max: {inst.max}</span>
                <span style={{ fontSize: 11, color: '#64748b', background: '#f1f5f9', padding: '2px 8px', borderRadius: 4 }}>Scoring: {inst.scoring}</span>
                <DirectionBadge direction={inst.direction} />
              </div>

              {inst.bands.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Interpretation Bands</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {inst.bands.map((b, bi) => (
                      <div key={bi} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <LevelBadge level={b.level} />
                        <span style={{ fontSize: 11, color: '#334155' }}>{b.label} ({b.min}–{b.max})</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {inst.domains && inst.domains.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Domains</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {inst.domains.map((d, di) => (
                      <span key={di} style={{ fontSize: 11, background: '#ede9fe', color: '#7c3aed', padding: '2px 8px', borderRadius: 4 }}>
                        {d.label} (max {d.max})
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {inst.items && inst.items.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Items ({inst.items.length})</div>
                  <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6 }}>
                    {inst.items.join(' · ')}
                  </div>
                </div>
              )}

              {inst.note && (
                <div style={{ fontSize: 11, color: '#3b82f6', background: '#eff6ff', padding: '6px 10px', borderRadius: 4, marginBottom: 4 }}>
                  {inst.note}
                </div>
              )}
              {inst.alert && (
                <div style={{ fontSize: 11, color: '#ef4444', background: '#fef2f2', padding: '6px 10px', borderRadius: 4 }}>
                  {inst.alert}
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* ── By Role Tab ── */
  const renderByRole = () => (
    <div>
      {roles.map(role => {
        const roleInsts = allBd.filter(i => i.role === role)
        return (
          <Card key={role} title={`${role} (${roleInsts.length} instrument${roleInsts.length !== 1 ? 's' : ''})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>ID</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Max</th>
                    <th style={thStyle}>Scoring</th>
                    <th style={thStyle}>Direction</th>
                    <th style={thStyle}>Bands</th>
                    <th style={thStyle}>Items</th>
                  </tr>
                </thead>
                <tbody>
                  {roleInsts.map(inst => (
                    <tr key={inst.id}>
                      <td style={tdStyle}><strong>{inst.icon} {inst.id}</strong></td>
                      <td style={tdStyle}>{inst.name}</td>
                      <td style={tdStyle}>{inst.max}</td>
                      <td style={tdStyle}>{inst.scoring}</td>
                      <td style={tdStyle}><DirectionBadge direction={inst.direction} /></td>
                      <td style={tdStyle}>{inst.bands.length}</td>
                      <td style={tdStyle}>{inst.items ? inst.items.length : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )
      })}
    </div>
  )

  /* ── Bands Tab ── */
  const renderBands = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
      <Card title="Band Distribution by Level" span={2}>
        {(() => {
          const levelCounts = {}
          allBd.forEach(inst => {
            inst.bands.forEach(b => {
              levelCounts[b.level] = (levelCounts[b.level] || 0) + 1
            })
          })
          const data = Object.entries(levelCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([level, count]) => ({ name: level, value: count }))
          return (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {data.map((d, i) => (
                    <Cell key={i} fill={LEVEL_COLORS[d.name] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )
        })()}
      </Card>

      {allBd.map(inst => (
        <Card key={inst.id} title={`${inst.icon} ${inst.id} — Score Range 0–${inst.max}`}>
          <div style={{ marginBottom: 8 }}>
            <DirectionBadge direction={inst.direction} />
          </div>
          <div style={{ display: 'flex', width: '100%', height: 24, borderRadius: 6, overflow: 'hidden', marginBottom: 8 }}>
            {inst.bands.map((b, bi) => {
              const range = inst.max > 0 ? ((b.max - b.min + 1) / (inst.max + 1)) * 100 : 0
              return (
                <div key={bi} style={{
                  width: `${range}%`, background: LEVEL_COLORS[b.level] || '#94a3b8',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 9, color: '#fff', fontWeight: 600, minWidth: 20
                }}>
                  {b.min}-{b.max}
                </div>
              )
            })}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {inst.bands.map((b, bi) => (
              <div key={bi} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
                <LevelBadge level={b.level} />
                <span style={{ color: '#334155' }}>{b.label} ({b.min}–{b.max})</span>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )

  /* ── Definitions Tab ── */
  const renderDefinitions = () => {
    if (!defs || !defs.available) return <div style={{ color: '#f97316' }}>Definitions not loaded</div>
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        <Card title="Scoring Legend">
          {(defs.scoring_legend || []).map((s, i) => (
            <div key={i} style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: 12 }}>{s.key}</strong>
              <span style={{ fontSize: 11, color: '#64748b', marginLeft: 8 }}>{s.label}</span>
            </div>
          ))}
        </Card>

        <Card title="Direction Legend">
          {(defs.direction_legend || []).map((d, i) => (
            <div key={i} style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: 12 }}>{d.key.replace(/_/g, ' ')}</strong>
              <span style={{ fontSize: 11, color: '#64748b', marginLeft: 8 }}>{d.label}</span>
            </div>
          ))}
        </Card>

        <Card title="Severity Levels">
          {(defs.severity_colors || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <div style={{ width: 16, height: 16, borderRadius: 4, background: s.color }} />
              <span style={{ fontSize: 12, fontWeight: 600 }}>{s.level}</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{s.label}</span>
            </div>
          ))}
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
                  <td style={tdStyle}>{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Clinical Notes">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {(defs.clinical_notes || []).map((n, i) => (
              <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 6, lineHeight: 1.5 }}>{n}</li>
            ))}
          </ul>
        </Card>

        <Card title="References">
          {(defs.references || []).map((r, i) => (
            <div key={i} style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: 12 }}>{r.name}</strong>
              <div style={{ fontSize: 11, color: '#64748b' }}>{r.desc}</div>
            </div>
          ))}
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Assessment Instruments Catalog
      </h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
        {overview.title} — {k.total_instruments} instruments, {k.unique_roles} roles, {k.total_bands} interpretation bands
        {overview.updated_at && ` — updated ${overview.updated_at}`}
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: '1px solid',
            borderColor: tab === t ? '#3b82f6' : '#e2e8f0',
            background: tab === t ? '#3b82f6' : '#fff',
            color: tab === t ? '#fff' : '#475569',
            fontSize: 12, fontWeight: 600, cursor: 'pointer', textTransform: 'capitalize'
          }}>
            {t.replace(/-/g, ' ')}
          </button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'all-instruments' && renderAllInstruments()}
      {tab === 'by-role' && renderByRole()}
      {tab === 'bands' && renderBands()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
