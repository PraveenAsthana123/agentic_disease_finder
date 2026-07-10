import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#84cc16']
const PIE_COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'per_patient', label: 'Per-Patient' },
  { id: 'category_matrix', label: 'Category Matrix' },
  { id: 'missing_fields', label: 'Missing Fields' },
  { id: 'definitions', label: 'Definitions' },
]

const qualityColor = (pct) => {
  if (pct >= 90) return '#10b981'
  if (pct >= 75) return '#3b82f6'
  if (pct >= 50) return '#f59e0b'
  return '#ef4444'
}

const qualityLabel = (pct) => {
  if (pct >= 90) return 'Excellent'
  if (pct >= 75) return 'Good'
  if (pct >= 50) return 'Fair'
  return 'Poor'
}

const HEATMAP_COLOR = (value) => {
  if (value == null) return '#f1f5f9'
  if (value >= 90) return '#dcfce7'
  if (value >= 75) return '#dbeafe'
  if (value >= 50) return '#fef3c7'
  if (value >= 25) return '#fed7aa'
  return '#fee2e2'
}

export default function DataCompletenessDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedPatient, setExpandedPatient] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/data-completeness/overview`),
      axios.get(`${API_URL}/api/data-completeness/breakdown`),
      axios.get(`${API_URL}/api/data-completeness/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Data Completeness Dashboard...</div>
  if (error) return <div style={{ padding: 20, color: '#ef4444', background: '#fef2f2', borderRadius: 8 }}>Error: {error}</div>

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Data Completeness Dashboard</h2>
      <p style={{ margin: '0 0 20px', color: '#64748b', fontSize: 13 }}>
        Per-patient data field completeness across 9 clinical data categories (real data from clinical.db)
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: tab === t.id ? '#eff6ff' : 'transparent', color: tab === t.id ? '#1d4ed8' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', borderRadius: '8px 8px 0 0', fontSize: 13,
            marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'per_patient' && renderPerPatient()}
      {tab === 'category_matrix' && renderCategoryMatrix()}
      {tab === 'missing_fields' && renderMissingFields()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    if (!overview?.available) return <div style={{ color: '#94a3b8' }}>No data available.</div>
    const { kpis, per_category, completeness_distribution, top_missing_fields } = overview

    const radarData = per_category.map(c => ({
      category: c.category.length > 12 ? c.category.substring(0, 12) + '...' : c.category,
      completeness: c.completeness_pct,
      fullName: c.category,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {/* KPI Cards */}
        <Card>
          <KPI label="Total Patients" value={kpis.total_patients} color="#3b82f6" />
        </Card>
        <Card>
          <KPI label="Overall Completeness" value={`${kpis.overall_completeness_pct}%`}
            color={qualityColor(kpis.overall_completeness_pct)}
            sub={qualityLabel(kpis.overall_completeness_pct)} />
        </Card>
        <Card>
          <KPI label="Total Fields Tracked" value={kpis.total_fields} color="#8b5cf6" />
        </Card>

        {/* Category Completeness Bar Chart */}
        <Card title="Completeness by Category" span={2}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={per_category} layout="vertical" margin={{ left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="category" width={110} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Bar dataKey="completeness_pct" name="Completeness %" radius={[0, 4, 4, 0]}>
                {per_category.map((entry, i) => (
                  <Cell key={i} fill={qualityColor(entry.completeness_pct)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Distribution Pie */}
        <Card title="Patient Distribution">
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={completeness_distribution} dataKey="count" nameKey="range" cx="50%" cy="50%"
                outerRadius={100} label={({ range, count }) => `${range}: ${count}`}>
                {completeness_distribution.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        {/* Radar Chart */}
        <Card title="Category Completeness Radar" span={2}>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={110}>
              <PolarGrid />
              <PolarAngleAxis dataKey="category" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
              <Radar name="Completeness %" dataKey="completeness" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              <Tooltip formatter={(v) => `${v}%`} />
            </RadarChart>
          </ResponsiveContainer>
        </Card>

        {/* Top Missing Fields */}
        <Card title="Top Missing Fields">
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            {(top_missing_fields || []).slice(0, 10).map((f, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                <div>
                  <div style={{ fontSize: 13, color: '#1e293b' }}>{f.field}</div>
                  <div style={{ fontSize: 10, color: '#94a3b8' }}>{f.category}</div>
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#ef4444' }}>
                  {f.missing_count} ({f.missing_pct}%)
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  function renderPerPatient() {
    if (!breakdown?.available) return <div style={{ color: '#94a3b8' }}>No data available.</div>
    const { per_patient } = breakdown

    return (
      <Card title={`Per-Patient Completeness (${per_patient.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient ID</th>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Fields Present</th>
                <th style={thStyle}>Completeness</th>
                <th style={thStyle}>Quality</th>
                <th style={thStyle}>Missing</th>
                <th style={thStyle}></th>
              </tr>
            </thead>
            <tbody>
              {per_patient.map((p, i) => (
                <React.Fragment key={p.patient_id}>
                  <tr style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafbfc' }}>
                    <td style={tdStyle}><span style={{ fontFamily: 'monospace', fontSize: 12 }}>{p.patient_id}</span></td>
                    <td style={tdStyle}>{p.name}</td>
                    <td style={tdStyle}>{p.fields_present}/{p.fields_total}</td>
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ width: 80, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                          <div style={{ width: `${p.completeness_pct}%`, height: '100%',
                            background: qualityColor(p.completeness_pct), borderRadius: 4 }} />
                        </div>
                        <span style={{ fontSize: 12, fontWeight: 600, color: qualityColor(p.completeness_pct) }}>
                          {p.completeness_pct}%
                        </span>
                      </div>
                    </td>
                    <td style={tdStyle}>
                      <span style={{ padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                        background: qualityColor(p.completeness_pct) + '20',
                        color: qualityColor(p.completeness_pct) }}>
                        {qualityLabel(p.completeness_pct)}
                      </span>
                    </td>
                    <td style={tdStyle}>{p.missing_count} fields</td>
                    <td style={tdStyle}>
                      <button onClick={() => setExpandedPatient(expandedPatient === p.patient_id ? null : p.patient_id)}
                        style={{ border: 'none', background: '#eff6ff', color: '#3b82f6', padding: '4px 10px',
                          borderRadius: 6, cursor: 'pointer', fontSize: 11 }}>
                        {expandedPatient === p.patient_id ? 'Hide' : 'Details'}
                      </button>
                    </td>
                  </tr>
                  {expandedPatient === p.patient_id && (
                    <tr>
                      <td colSpan={7} style={{ padding: '12px 20px', background: '#f8fafc' }}>
                        <div style={{ fontSize: 12, color: '#334155', marginBottom: 6, fontWeight: 600 }}>Missing Fields:</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                          {p.missing_fields.map((f, j) => (
                            <span key={j} style={{ padding: '3px 10px', background: '#fee2e2', color: '#991b1b',
                              borderRadius: 12, fontSize: 11 }}>{f}</span>
                          ))}
                          {p.missing_fields.length === 0 && (
                            <span style={{ color: '#10b981', fontSize: 12 }}>All fields present</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )
  }

  function renderCategoryMatrix() {
    if (!breakdown?.available) return <div style={{ color: '#94a3b8' }}>No data available.</div>
    const { category_matrix, category_rankings } = breakdown
    const categories = category_rankings.map(c => c.category)

    return (
      <div style={{ display: 'grid', gap: 16 }}>
        {/* Rankings */}
        <Card title="Category Rankings (by average completeness)">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
            {category_rankings.map((c, i) => (
              <div key={c.category} style={{ padding: 12, background: '#f8fafc', borderRadius: 8,
                borderLeft: `4px solid ${qualityColor(c.completeness_pct)}` }}>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>#{c.rank}</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b', marginTop: 2 }}>{c.category}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: qualityColor(c.completeness_pct), marginTop: 4 }}>
                  {c.completeness_pct}%
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Heatmap Table */}
        <Card title="Patient x Category Completeness Matrix">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
              <thead>
                <tr>
                  <th style={{ ...thStyle, fontSize: 11, position: 'sticky', left: 0, background: '#f8fafc', zIndex: 1 }}>Patient</th>
                  {categories.map(c => (
                    <th key={c} style={{ ...thStyle, fontSize: 10, writingMode: 'vertical-rl', textOrientation: 'mixed',
                      height: 100, whiteSpace: 'nowrap' }}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {category_matrix.map((row, i) => (
                  <tr key={row.patient_id}>
                    <td style={{ ...tdStyle, fontSize: 11, fontFamily: 'monospace', position: 'sticky', left: 0,
                      background: i % 2 === 0 ? '#fff' : '#fafbfc', zIndex: 1 }}>
                      {row.patient_id}
                    </td>
                    {categories.map(cat => {
                      const val = row[cat]
                      return (
                        <td key={cat} style={{ ...tdStyle, textAlign: 'center', background: HEATMAP_COLOR(val),
                          fontWeight: 600, fontSize: 11, color: val >= 50 ? '#1e293b' : '#991b1b' }}>
                          {val}%
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderMissingFields() {
    if (!overview?.available) return <div style={{ color: '#94a3b8' }}>No data available.</div>
    const { top_missing_fields } = overview

    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title="Missing Fields Ranked by Impact">
          <ResponsiveContainer width="100%" height={Math.max(300, top_missing_fields.length * 28)}>
            <BarChart data={top_missing_fields} layout="vertical" margin={{ left: 150 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="field" width={140} tick={{ fontSize: 11 }} />
              <Tooltip content={({ payload }) => {
                if (!payload || !payload.length) return null
                const d = payload[0].payload
                return (
                  <div style={{ background: '#fff', padding: 10, borderRadius: 8, boxShadow: '0 2px 8px rgba(0,0,0,.15)',
                    fontSize: 12 }}>
                    <div style={{ fontWeight: 600 }}>{d.field}</div>
                    <div style={{ color: '#64748b' }}>Category: {d.category}</div>
                    <div style={{ color: '#ef4444' }}>Missing: {d.missing_count} patients ({d.missing_pct}%)</div>
                    <div style={{ color: '#10b981' }}>Present: {d.present_count} patients</div>
                  </div>
                )
              }} />
              <Bar dataKey="missing_count" name="Missing Count" radius={[0, 4, 4, 0]}>
                {top_missing_fields.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Detailed table */}
        <Card title="Missing Fields Detail">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Field</th>
                <th style={thStyle}>Category</th>
                <th style={thStyle}>Missing Count</th>
                <th style={thStyle}>Missing %</th>
                <th style={thStyle}>Present Count</th>
                <th style={thStyle}>Impact</th>
              </tr>
            </thead>
            <tbody>
              {top_missing_fields.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={tdStyle}>{f.field}</td>
                  <td style={tdStyle}><span style={{ padding: '2px 8px', background: '#eff6ff', color: '#1d4ed8',
                    borderRadius: 12, fontSize: 11 }}>{f.category}</span></td>
                  <td style={{ ...tdStyle, color: '#ef4444', fontWeight: 600 }}>{f.missing_count}</td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: 60, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${f.missing_pct}%`, height: '100%', background: '#ef4444', borderRadius: 3 }} />
                      </div>
                      {f.missing_pct}%
                    </div>
                  </td>
                  <td style={tdStyle}>{f.present_count}</td>
                  <td style={tdStyle}>
                    <span style={{ padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                      background: f.missing_pct >= 75 ? '#fee2e2' : f.missing_pct >= 50 ? '#fef3c7' : '#dcfce7',
                      color: f.missing_pct >= 75 ? '#991b1b' : f.missing_pct >= 50 ? '#92400e' : '#166534'
                    }}>
                      {f.missing_pct >= 75 ? 'Critical' : f.missing_pct >= 50 ? 'High' : f.missing_pct >= 25 ? 'Medium' : 'Low'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions) return <div style={{ color: '#94a3b8' }}>No definitions available.</div>

    return (
      <div style={{ display: 'grid', gap: 16 }}>
        {/* Quality Levels */}
        <Card title="Data Quality Levels">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            {(definitions.data_quality_levels || []).map((l, i) => (
              <div key={i} style={{ padding: 16, borderRadius: 8, background: '#f8fafc',
                borderLeft: `4px solid ${l.color}` }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: l.color }}>{l.level}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{l.range}</div>
                <div style={{ fontSize: 11, color: '#334155', marginTop: 8 }}>{l.description}</div>
              </div>
            ))}
          </div>
        </Card>

        {/* Category Definitions */}
        <Card title="Category Definitions">
          {(definitions.categories || []).map((cat, i) => (
            <div key={i} style={{ padding: 14, borderBottom: '1px solid #f1f5f9' }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{cat.name}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{cat.description}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                {cat.fields.map((f, j) => (
                  <span key={j} style={{ padding: '3px 10px', background: '#eff6ff', color: '#1d4ed8',
                    borderRadius: 12, fontSize: 11 }}>{f}</span>
                ))}
              </div>
            </div>
          ))}
        </Card>

        {/* Methodology */}
        <Card title="Methodology">
          <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6, margin: 0 }}>
            {definitions.methodology}
          </p>
        </Card>
      </div>
    )
  }
}

const thStyle = { textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600, fontSize: 12,
  borderBottom: '2px solid #e2e8f0' }
const tdStyle = { padding: '8px 12px', color: '#334155' }
