import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { present: '#4caf50', partial: '#ff9800', missing: '#f44336' }

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

export default function DataRequirementsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/data-requirements/overview`),
          axios.get(`${API_URL}/api/data-requirements/breakdown`),
          axios.get(`${API_URL}/api/data-requirements/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Data Requirements data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading data requirements...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Data requirements data not available.'}
    </div>
  )

  const kpis = overview.kpis || {}
  const catBreakdown = overview.category_breakdown || []
  const statusDist = overview.status_distribution || {}
  const tierCoverage = overview.tier_coverage || []
  const cgSummary = overview.control_groups_summary || {}

  const allItems = breakdown?.all_items || []
  const artifactTemplate = breakdown?.artifact_template || []
  const techDeliverables = breakdown?.technician_deliverables || []
  const top10 = breakdown?.top10_artifacts || []
  const controlGroups = breakdown?.control_groups || {}

  const statusPie = [
    { name: 'Present', value: statusDist.present || 0 },
    { name: 'Partial', value: statusDist.partial || 0 },
    { name: 'Missing', value: statusDist.missing || 0 },
  ].filter(d => d.value > 0)
  const pieColors = ['#4caf50', '#ff9800', '#f44336']

  const statusLevels = defs?.status_levels || []
  const dataTiers = defs?.data_tiers || []
  const glossary = defs?.glossary || []
  const clinicalNotes = defs?.clinical_notes || []
  const references = defs?.references || []

  const tabs = ['overview', 'breakdown', 'definitions']
  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', borderRadius: '6px 6px 0 0', fontSize: 13, fontWeight: 600,
    background: tab === t ? '#1e88e5' : '#f1f5f9', color: tab === t ? '#fff' : '#475569',
    border: tab === t ? '1px solid #1e88e5' : '1px solid #e2e8f0', borderBottom: 'none'
  })

  const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', background: '#f8fafc', fontSize: 12, color: '#475569' }
  const tdStyle = { padding: '6px 10px', fontSize: 12, color: '#334155', borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Data Requirements Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 12, color: '#64748b' }}>{overview.note}</p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => <div key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t.charAt(0).toUpperCase() + t.slice(1)}</div>)}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          {/* KPI Row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {[
              { label: 'Total Items', value: kpis.total_items, color: COLORS[3] },
              { label: 'Present', value: kpis.present, color: COLORS[0] },
              { label: 'Partial', value: kpis.partial, color: COLORS[1] },
              { label: 'Missing', value: kpis.missing, color: COLORS[2] },
              { label: 'Completeness %', value: `${kpis.completeness_pct}%`, color: COLORS[0] },
              { label: 'Categories', value: kpis.categories, color: COLORS[5] },
              { label: 'Tier 1 Mandatory', value: kpis.tier1_mandatory, color: COLORS[4] },
              { label: 'Control Groups', value: kpis.control_groups, color: COLORS[7] },
            ].map((k, i) => (
              <div key={i} style={kpiStyle(k.color)}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: k.color }}>{fmt(k.value)}</div>
              </div>
            ))}
          </div>

          {/* Single Most Important Note */}
          {overview.single_most_important && (
            <div style={{ ...cardStyle, background: '#fef3c7', border: '1px solid #fbbf24', padding: '12px 16px' }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: '#92400e' }}>Most Critical: </span>
              <span style={{ fontSize: 12, color: '#78350f' }}>{overview.single_most_important}</span>
            </div>
          )}

          {/* Charts */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 18, marginBottom: 18 }}>
            {/* Status Distribution Pie */}
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Status Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={statusPie} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" outerRadius={80}
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {statusPie.map((_, i) => <Cell key={i} fill={pieColors[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Category Stacked Horizontal Bar */}
            <div style={{ ...cardStyle, gridColumn: 'span 1' }}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Category Status Breakdown</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  layout="vertical"
                  data={catBreakdown.map(c => ({
                    name: c.category.replace(/^\d+\.\s*/, '').slice(0, 22),
                    Present: c.present,
                    Partial: c.partial,
                    Missing: c.missing,
                  }))}
                  margin={{ left: 8, right: 8, top: 4, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" fontSize={10} />
                  <YAxis type="category" dataKey="name" fontSize={9} width={110} />
                  <Tooltip />
                  <Bar dataKey="Present" stackId="a" fill="#4caf50" />
                  <Bar dataKey="Partial" stackId="a" fill="#ff9800" />
                  <Bar dataKey="Missing" stackId="a" fill="#f44336" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Tier Coverage Table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Tier Coverage</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={thStyle}>Tier</th>
                  <th style={{ ...thStyle, textAlign: 'center' }}>Items</th>
                </tr>
              </thead>
              <tbody>
                {tierCoverage.map((t, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{t.label}</td>
                    <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 700, color: COLORS[3] }}>{t.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Control Groups Summary */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Control Groups</h4>
            {cgSummary.note && (
              <p style={{ margin: '0 0 10px', fontSize: 12, color: '#475569' }}>{cgSummary.note}</p>
            )}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {(cgSummary.most_valuable || []).map((cg, i) => (
                <span key={i} style={{
                  background: `${COLORS[i % COLORS.length]}11`,
                  border: `1px solid ${COLORS[i % COLORS.length]}44`,
                  borderRadius: 6, padding: '4px 12px', fontSize: 12, color: '#334155', fontWeight: 500
                }}>{cg}</span>
              ))}
            </div>
            <div style={{ marginTop: 10, fontSize: 12, color: '#64748b' }}>
              Minimum dataset: <strong>{cgSummary.minimum_cohorts}</strong> cohorts &nbsp;|&nbsp;
              Ideal dataset: <strong>{cgSummary.ideal_cohorts}</strong> cohorts
            </div>
          </div>
        </>
      )}

      {/* BREAKDOWN TAB */}
      {tab === 'breakdown' && (
        <>
          {/* All Items Table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>All Data Items ({allItems.length})</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Name</th>
                    <th style={{ ...thStyle, textAlign: 'center' }}>Status</th>
                    <th style={thStyle}>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {allItems.map((item, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ ...tdStyle, color: '#64748b', fontSize: 11 }}>{item.category.replace(/^\d+\.\s*/, '')}</td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{item.name}</td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}><Badge status={item.status} /></td>
                      <td style={{ ...tdStyle, color: '#64748b' }}>{item.note || ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Artifact Template Table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Artifact Template ({artifactTemplate.length} types)</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Type</th>
                    <th style={{ ...thStyle, textAlign: 'center' }}>Mandatory</th>
                  </tr>
                </thead>
                <tbody>
                  {artifactTemplate.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={tdStyle}>{a.category}</td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{a.type}</td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>
                        <span style={{ color: a.mandatory ? '#4caf50' : '#94a3b8', fontWeight: 700 }}>
                          {a.mandatory ? 'Yes' : 'No'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Technician Deliverables Table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Technician Deliverables ({techDeliverables.length} items)</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Item</th>
                    <th style={thStyle}>Format</th>
                  </tr>
                </thead>
                <tbody>
                  {techDeliverables.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{d.item}</td>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', color: '#1e88e5' }}>{d.format}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Top 10 Artifacts */}
          {top10.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Top 10 Artifacts to Label</h4>
              <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
                {top10.map((a, i) => <li key={i} style={{ marginBottom: 4 }}>{a}</li>)}
              </ol>
            </div>
          )}

          {/* Control Group Dataset Tables */}
          {controlGroups.minimum_dataset && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 18 }}>
              <div style={cardStyle}>
                <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Minimum Dataset</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Cohort</th>
                      <th style={{ ...thStyle, textAlign: 'center' }}>n</th>
                    </tr>
                  </thead>
                  <tbody>
                    {controlGroups.minimum_dataset.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={tdStyle}>{r.cohort}</td>
                        <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 700 }}>{r.n}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={cardStyle}>
                <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Ideal Dataset</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Cohort</th>
                      <th style={{ ...thStyle, textAlign: 'center' }}>n</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(controlGroups.ideal_dataset || []).map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={tdStyle}>{r.cohort}</td>
                        <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 700 }}>{r.n}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <>
          {/* Status Levels */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Status Levels</h4>
            {statusLevels.map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 12, paddingBottom: 12, borderBottom: i < statusLevels.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <span style={{ background: `${s.color}22`, color: s.color, border: `1px solid ${s.color}55`, borderRadius: 4, padding: '2px 10px', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', whiteSpace: 'nowrap' }}>{s.label}</span>
                <span style={{ fontSize: 12, color: '#475569' }}>{s.description}</span>
              </div>
            ))}
          </div>

          {/* Data Tiers */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Data Tiers</h4>
            {dataTiers.map((t, i) => (
              <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < dataTiers.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{t.label}</div>
                <div style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{t.description}</div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {(t.items || []).map((item, j) => (
                    <span key={j} style={{ background: '#f1f5f9', border: '1px solid #e2e8f0', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#475569' }}>{item}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Glossary */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Glossary ({glossary.length} terms)</h4>
            {glossary.map((g, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8, paddingBottom: 8, borderBottom: i < glossary.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <span style={{ fontFamily: 'monospace', fontWeight: 700, color: '#1e88e5', minWidth: 60, fontSize: 12 }}>{g.term}</span>
                <span style={{ fontSize: 12, color: '#475569' }}>{g.definition}</span>
              </div>
            ))}
          </div>

          {/* Clinical Notes */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Clinical Notes</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569' }}>
              {clinicalNotes.map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
            </ul>
          </div>

          {/* References */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>References</h4>
            <ol style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569' }}>
              {references.map((r, i) => <li key={i} style={{ marginBottom: 6 }}>{r}</li>)}
            </ol>
          </div>
        </>
      )}
    </div>
  )
}
