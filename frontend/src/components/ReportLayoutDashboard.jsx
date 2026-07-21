import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#eab308', '#14b8a6', '#f43f5e', '#a855f7']
const STATUS_COLORS = { built: '#22c55e', planned: '#f97316', unknown: '#94a3b8' }

const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 13, fontWeight: 600, color: '#475569', whiteSpace: 'nowrap' }
const tdStyle = { padding: '8px 12px', borderBottom: '1px solid #f1f5f9', fontSize: 13, color: '#334155' }

function Card({ title, children, span }) {
  return (
    <div style={{ background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)', gridColumn: span ? `span ${span}` : undefined }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{status}</span>
}

function TypeBadge({ type }) {
  const typeColors = { AI: '#3b82f6', Expert: '#8b5cf6', Final: '#22c55e', Audit: '#f97316', Other: '#94a3b8' }
  const color = typeColors[type] || '#94a3b8'
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{type}</span>
}

function EditableBadge({ editable }) {
  const color = editable ? '#22c55e' : '#94a3b8'
  const label = editable ? 'Editable' : 'Read-only'
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{label}</span>
}

export default function ReportLayoutDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/report-layout/overview`),
      axios.get(`${API}/report-layout/breakdown`),
      axios.get(`${API}/report-layout/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Report Layout...</div>
  if (overview.available === false) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'components', 'sections', 'report-types', 'definitions']
  const k = overview.kpis || {}

  /* -- Overview Tab -- */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
          <KPI label="Report Types" value={k.total_report_types} />
          <KPI label="Components" value={k.total_components} />
          <KPI label="Sections" value={k.total_sections} />
          <KPI label="Editable Sections" value={k.editable_sections} />
          <KPI label="AI Sources" value={k.total_ai_sources} />
        </div>
      </Card>

      <Card title="Components by AI Source">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.components_by_source || []} layout="vertical" margin={{ left: 140, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Components" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Section Type Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.section_type_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
              {(overview.section_type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Editable vs Read-only">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.editability_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
              {(overview.editability_distribution || []).map((_, i) => <Cell key={i} fill={[COLORS[1], COLORS[4]][i] || COLORS[i]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Components Summary" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>ID</th>
              <th style={thStyle}>Label</th>
              <th style={thStyle}>AI Source</th>
              <th style={thStyle}>AI Finding</th>
            </tr></thead>
            <tbody>
              {(overview.components_table || []).map((c, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{c.id}</td>
                  <td style={tdStyle}>{c.label}</td>
                  <td style={tdStyle}><span style={{ background: '#e0e7ff', color: '#3730a3', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 500 }}>{c.ai_source}</span></td>
                  <td style={{ ...tdStyle, fontSize: 11, color: '#64748b', maxWidth: 300 }}>{c.ai_finding}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Status" span={2}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {Object.entries(k.status_summary || {}).map(([key, val], i) => (
            <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: '10px 16px', border: '1px solid #e2e8f0', flex: '1 1 200px' }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>{key.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 13, color: '#1e293b' }}>{val}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )

  /* -- Components Tab -- */
  const renderComponents = () => {
    const bd = breakdown || { per_component: [] }
    const comps = bd.per_component || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Report Components (${comps.length})`}>
          <div style={{ maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>ID</th>
                <th style={thStyle}>Label</th>
                <th style={thStyle}>AI Source</th>
                <th style={thStyle}>AI Finding</th>
                <th style={thStyle}>AI Recommendation</th>
              </tr></thead>
              <tbody>
                {comps.map((c, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{c.id}</td>
                    <td style={tdStyle}>{c.label}</td>
                    <td style={tdStyle}><span style={{ background: '#e0e7ff', color: '#3730a3', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 500 }}>{c.ai_source}</span></td>
                    <td style={{ ...tdStyle, fontSize: 12, color: '#64748b', maxWidth: 300 }}>{c.ai_finding}</td>
                    <td style={{ ...tdStyle, fontSize: 12, color: '#64748b', maxWidth: 300 }}>{c.ai_recommendation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 16 }}>
          {comps.map((c, i) => (
            <Card key={i} title={c.label}>
              <div style={{ marginBottom: 8 }}>
                <span style={{ background: '#e0e7ff', color: '#3730a3', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 500 }}>{c.ai_source}</span>
              </div>
              <div style={{ background: '#f8fafc', borderRadius: 6, padding: 10, border: '1px solid #e2e8f0', marginBottom: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>AI Finding</div>
                <div style={{ fontSize: 12, color: '#334155' }}>{c.ai_finding}</div>
              </div>
              <div style={{ background: '#f0fdf4', borderRadius: 6, padding: 10, border: '1px solid #bbf7d0' }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#166534', marginBottom: 4 }}>AI Recommendation</div>
                <div style={{ fontSize: 12, color: '#334155' }}>{c.ai_recommendation}</div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* -- Sections Tab -- */
  const renderSections = () => {
    const bd = breakdown || { per_section_type: [] }
    const sectionTypes = bd.per_section_type || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Report Sections (${(overview.sections_table || []).length})`}>
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>ID</th>
                <th style={thStyle}>Label</th>
                <th style={thStyle}>Source</th>
                <th style={thStyle}>Type</th>
                <th style={thStyle}>Editable</th>
              </tr></thead>
              <tbody>
                {(overview.sections_table || []).map((s, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{s.id}</td>
                    <td style={tdStyle}>{s.label}</td>
                    <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>{s.source}</td>
                    <td style={tdStyle}><TypeBadge type={s.type} /></td>
                    <td style={tdStyle}><EditableBadge editable={s.editable} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
          {sectionTypes.map((st, i) => (
            <Card key={i} title={`${st.type} Sections`}>
              <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 11, color: '#64748b' }}>{st.sections.length} section{st.sections.length !== 1 ? 's' : ''}</span>
              </div>
              {st.sections.map((s, si) => (
                <div key={si} style={{ background: '#f8fafc', borderRadius: 6, padding: 10, border: '1px solid #e2e8f0', marginBottom: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{s.label}</span>
                    <EditableBadge editable={s.editable} />
                  </div>
                  <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}>Source: {s.source}</p>
                </div>
              ))}
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* -- Report Types Tab -- */
  const renderReportTypes = () => {
    const bd = breakdown || { report_type_details: [], status_items: [] }
    const rtDetails = bd.report_type_details || []
    const statusItems = bd.status_items || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title={`Report Types (${rtDetails.length})`} span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {rtDetails.map((rt, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 6 }}>{rt.name}</div>
                <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>{rt.description}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Implementation Status" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {statusItems.map((si, i) => {
              const isBuilt = si.value.toLowerCase().includes('built')
              const isPlanned = si.value.toLowerCase().includes('planned')
              return (
                <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: `1px solid ${isBuilt ? '#bbf7d0' : isPlanned ? '#fed7aa' : '#e2e8f0'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{si.key.replace(/_/g, ' ')}</span>
                    <StatusBadge status={isBuilt ? 'built' : isPlanned ? 'planned' : 'unknown'} />
                  </div>
                  <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>{si.value}</p>
                </div>
              )
            })}
          </div>
        </Card>
      </div>
    )
  }

  /* -- Definitions Tab -- */
  const renderDefinitions = () => {
    const d = defs || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Status Legend">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={thStyle}>Status</th><th style={thStyle}>Description</th></tr></thead>
            <tbody>
              {(d.status_legend || []).map((s, i) => (
                <tr key={i}><td style={tdStyle}><StatusBadge status={s.status} /></td><td style={tdStyle}>{s.description}</td></tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Glossary" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr></thead>
            <tbody>
              {(d.glossary || []).map((g, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                  <td style={tdStyle}>{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {d.clinical_notes && d.clinical_notes.length > 0 && (
          <Card title="Clinical Notes" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {d.clinical_notes.map((n, i) => <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{n}</li>)}
            </ul>
          </Card>
        )}

        {d.references && d.references.length > 0 && (
          <Card title="References" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {d.references.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>
                  <strong>{r.ref}</strong> — {r.detail}
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Report Layout</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        {k.total_report_types} report types, {k.total_components} components, {k.total_sections} sections, {k.total_ai_sources} AI sources — EEG/Video-EEG summary report structure
      </p>

      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '2px solid #e2e8f0' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', background: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t ? 600 : 400,
            color: tab === t ? '#3b82f6' : '#64748b',
            borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent',
            marginBottom: -2, textTransform: 'capitalize'
          }}>{t.replace(/-/g, ' ')}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'components' && renderComponents()}
      {tab === 'sections' && renderSections()}
      {tab === 'report-types' && renderReportTypes()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
