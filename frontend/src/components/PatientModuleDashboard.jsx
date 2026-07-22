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

function TierBadge({ tier }) {
  const colors = { 1: '#ef4444', 2: '#f97316', 3: '#8b5cf6' }
  const labels = { 1: 'Tier 1 — Mandatory', 2: 'Tier 2 — Recommended', 3: 'Tier 3 — DBA Excellent' }
  const bg = colors[tier] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {labels[tier] || `Tier ${tier}`}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function PatientModuleDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/patient-module/overview`),
      axios.get(`${API_URL}/api/patient-module/breakdown`),
      axios.get(`${API_URL}/api/patient-module/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Patient Module...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'sections', 'tiers', 'control-groups', 'definitions']
  const k = overview.kpis || {}
  const charts = overview.charts || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Patient Module KPIs" span={2}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'space-around' }}>
          <KPI label="Sections" value={k.total_sections} />
          <KPI label="Total Fields" value={k.total_fields} />
          <KPI label="Built" value={k.built} sub="sections" />
          <KPI label="Tier 1 Items" value={k.tier1_count} sub="mandatory" />
          <KPI label="Tier 2 Items" value={k.tier2_count} sub="recommended" />
          <KPI label="Tier 3 Items" value={k.tier3_count} sub="DBA excellent" />
          <KPI label="Control Groups" value={k.control_groups} sub="differential" />
          <KPI label="Min Cohort N" value={k.min_cohort_total} />
        </div>
      </Card>

      {charts.status_distribution && charts.status_distribution.length > 0 && (
        <Card title="Section Status Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={charts.status_distribution} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                {charts.status_distribution.map((_, i) => (
                  <Cell key={i} fill={STATUS_COLORS[charts.status_distribution[i].name.toLowerCase()] || COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      )}

      <Card title="Fields per Section (estimated midpoint)">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={charts.fields_per_section} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v, _, p) => [`${v} (range: ${p.payload.range})`, 'Fields']} />
            <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Items per Section">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={charts.items_per_section} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="All Sections Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>#</th>
                <th style={thStyle}>Section</th>
                <th style={thStyle}>Fields</th>
                <th style={thStyle}>Items</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Note</th>
              </tr>
            </thead>
            <tbody>
              {(overview.sections_table || []).map((s, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={tdStyle}>{s.n}</td>
                  <td style={{ ...tdStyle, fontWeight: 500 }}>{s.section}</td>
                  <td style={tdStyle}>{s.fields}</td>
                  <td style={tdStyle}>{s.items_count}</td>
                  <td style={tdStyle}><StatusBadge status={s.status} /></td>
                  <td style={{ ...tdStyle, fontSize: 11, color: '#64748b', maxWidth: 300 }}>{s.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  const renderSections = () => {
    if (!breakdown || !breakdown.sections) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
        {breakdown.sections.map((s, i) => (
          <Card key={i} title={`${s.n}. ${s.section}`}>
            <div style={{ display: 'flex', gap: 12, marginBottom: 10, alignItems: 'center' }}>
              <StatusBadge status={s.status} />
              <span style={{ fontSize: 12, color: '#64748b' }}>{s.fields} fields</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
              {(s.items || []).map((item, j) => (
                <span key={j} style={{
                  background: '#f1f5f9', borderRadius: 4, padding: '3px 8px',
                  fontSize: 11, color: '#475569'
                }}>
                  {item}
                </span>
              ))}
            </div>
            {s.note && <div style={{ fontSize: 11, color: '#64748b', fontStyle: 'italic', borderTop: '1px solid #f1f5f9', paddingTop: 8 }}>{s.note}</div>}
          </Card>
        ))}
      </div>
    )
  }

  const renderTiers = () => {
    if (!breakdown || !breakdown.tiers) return null
    const { tiers: t } = breakdown
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
        <Card title="Tier 1 — Mandatory">
          <div style={{ marginBottom: 8 }}><TierBadge tier={1} /></div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(t.tier1_mandatory || []).map((item, i) => (
              <span key={i} style={{
                background: '#fef2f2', borderRadius: 4, padding: '4px 10px',
                fontSize: 12, color: '#dc2626', border: '1px solid #fecaca'
              }}>
                {item}
              </span>
            ))}
          </div>
        </Card>
        <Card title="Tier 2 — Recommended">
          <div style={{ marginBottom: 8 }}><TierBadge tier={2} /></div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(t.tier2_recommended || []).map((item, i) => (
              <span key={i} style={{
                background: '#fff7ed', borderRadius: 4, padding: '4px 10px',
                fontSize: 12, color: '#ea580c', border: '1px solid #fed7aa'
              }}>
                {item}
              </span>
            ))}
          </div>
        </Card>
        <Card title="Tier 3 — DBA Excellent">
          <div style={{ marginBottom: 8 }}><TierBadge tier={3} /></div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(t.tier3_dba_excellent || []).map((item, i) => (
              <span key={i} style={{
                background: '#f5f3ff', borderRadius: 4, padding: '4px 10px',
                fontSize: 12, color: '#7c3aed', border: '1px solid #ddd6fe'
              }}>
                {item}
              </span>
            ))}
          </div>
        </Card>

        {breakdown.single_most_important && (
          <Card title="Single Most Important Data Item" span={2}>
            <div style={{
              background: '#fef3c7', border: '1px solid #fbbf24', borderRadius: 6,
              padding: 12, fontSize: 13, color: '#92400e', fontWeight: 500
            }}>
              {breakdown.single_most_important}
            </div>
          </Card>
        )}

        {breakdown.technician_deliverables && breakdown.technician_deliverables.length > 0 && (
          <Card title="Technician Deliverables" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Item</th>
                    <th style={thStyle}>Format</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.technician_deliverables.map((d, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}>{d.item}</td>
                      <td style={tdStyle}><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 3, fontSize: 11 }}>{d.format}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    )
  }

  const renderControlGroups = () => {
    if (!breakdown || !breakdown.control_groups) return null
    const cg = breakdown.control_groups
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
        {cg.note && (
          <Card title="Control Group Rationale" span={2}>
            <div style={{
              background: '#eff6ff', border: '1px solid #93c5fd', borderRadius: 6,
              padding: 12, fontSize: 13, color: '#1e40af'
            }}>
              {cg.note}
            </div>
          </Card>
        )}

        <Card title="Most Valuable Control Groups">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(cg.most_valuable || []).map((g, i) => (
              <span key={i} style={{
                background: '#ecfdf5', borderRadius: 4, padding: '4px 10px',
                fontSize: 12, color: '#059669', border: '1px solid #a7f3d0'
              }}>
                {g}
              </span>
            ))}
          </div>
        </Card>

        <Card title="Minimum Dataset">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={charts.minimum_dataset}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ textAlign: 'center', fontSize: 11, color: '#64748b', marginTop: 4 }}>
            Total minimum N = {k.min_cohort_total}
          </div>
        </Card>

        <Card title="Ideal Dataset">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={charts.ideal_dataset}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={50} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ textAlign: 'center', fontSize: 11, color: '#64748b', marginTop: 4 }}>
            Total ideal N = {k.ideal_cohort_total}
          </div>
        </Card>

        {breakdown.artifact_template && breakdown.artifact_template.length > 0 && (
          <Card title="Artifact Template (12 mandatory types)" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Mandatory</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.artifact_template.map((a, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}>{a.category}</td>
                      <td style={tdStyle}>{a.type}</td>
                      <td style={tdStyle}>{a.mandatory ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!defs) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
        <Card title="Status Legend">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(defs.status_legend || []).map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{
                  background: `${s.color}22`, color: s.color, border: `1px solid ${s.color}55`,
                  borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', minWidth: 60, textAlign: 'center'
                }}>
                  {s.status}
                </span>
                <span style={{ fontSize: 12, color: '#475569' }}>{s.meaning}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Glossary" span={2}>
          <div style={{ overflowX: 'auto' }}>
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
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Clinical Notes">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {(defs.clinical_notes || []).map((n, i) => (
              <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{n}</li>
            ))}
          </ul>
        </Card>

        <Card title="References">
          {(defs.references || []).map((r, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>{r.label}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>{r.description}</div>
            </div>
          ))}
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: '16px 24px', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, fontWeight: 700, color: '#0f172a' }}>{overview.title || 'Patient Module Dashboard'}</h2>
      <p style={{ margin: '0 0 16px', fontSize: 12, color: '#64748b' }}>
        8-section, ~1,250-field patient data model — demographics through digital twin
      </p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 500, cursor: 'pointer',
              border: tab === t ? '2px solid #3b82f6' : '1px solid #e2e8f0',
              background: tab === t ? '#eff6ff' : '#fff',
              color: tab === t ? '#2563eb' : '#64748b'
            }}
          >
            {t.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
          </button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'sections' && renderSections()}
      {tab === 'tiers' && renderTiers()}
      {tab === 'control-groups' && renderControlGroups()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
