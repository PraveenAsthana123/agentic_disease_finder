import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
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

const STATUS_COLORS = { present: '#10b981', partial: '#f59e0b', missing: '#ef4444' }
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'category', label: 'Category Detail' },
  { id: 'gaps', label: 'Data Gaps' },
  { id: 'controls', label: 'Control Groups' },
  { id: 'definitions', label: 'Definitions' },
]

export default function DatasetRequirementsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/dataset-requirements/overview`),
      axios.get(`${API_URL}/api/dataset-requirements/breakdown`),
      axios.get(`${API_URL}/api/dataset-requirements/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Dataset Requirements data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Dataset Requirements Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Data completeness tracking — category breakdown, tier compliance, gap analysis, control group requirements
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'category' && breakdown && <CategoryDetailTab data={breakdown} />}
      {tab === 'gaps' && overview && <DataGapsTab data={overview} breakdown={breakdown} />}
      {tab === 'controls' && breakdown && <ControlGroupsTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 10, fontSize: 11, fontWeight: 600,
      background: `${color}18`, color: color, textTransform: 'capitalize'
    }}>
      {status}
    </span>
  )
}

function OverviewTab({ data }) {
  const statusPieData = [
    { name: 'Present', value: data.status_distribution.present },
    { name: 'Partial', value: data.status_distribution.partial },
    { name: 'Missing', value: data.status_distribution.missing },
  ]
  const pieColors = [STATUS_COLORS.present, STATUS_COLORS.partial, STATUS_COLORS.missing]

  const categoryChartData = (data.category_summary || []).map(c => ({
    category: c.category.replace(/^\d+\.\s*/, ''),
    present: c.present,
    partial: c.partial,
    missing: c.missing,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
      <Card title="Total Items"><KPI label="Dataset requirements" value={data.total_items} /></Card>
      <Card title="Present"><KPI label="Fully available" value={data.present_count} color="#10b981" /></Card>
      <Card title="Partial"><KPI label="Partially available" value={data.partial_count} color="#f59e0b" /></Card>
      <Card title="Missing"><KPI label="Not available" value={data.missing_count} color="#ef4444" /></Card>
      <Card title="Completeness"><KPI label="Overall score" value={`${data.overall_completeness}%`} color={data.overall_completeness >= 70 ? '#10b981' : data.overall_completeness >= 40 ? '#f59e0b' : '#ef4444'} /></Card>

      <Card title="Category Completeness (Present / Partial / Missing)" span={3}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={categoryChartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={12} />
            <YAxis dataKey="category" type="category" fontSize={10} width={140} />
            <Tooltip />
            <Legend />
            <Bar dataKey="present" fill={STATUS_COLORS.present} name="Present" stackId="a" />
            <Bar dataKey="partial" fill={STATUS_COLORS.partial} name="Partial" stackId="a" />
            <Bar dataKey="missing" fill={STATUS_COLORS.missing} name="Missing" stackId="a" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={statusPieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
              outerRadius={100} label={({ name, value }) => `${name} (${value})`} labelLine={false}
              fontSize={11}>
              {statusPieData.map((_, i) => (
                <Cell key={i} fill={pieColors[i]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Tier Compliance" span={5}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280, 1fr))', gap: 16 }}>
          {(data.tier_compliance || []).map((tier, i) => {
            const compColor = tier.compliance_pct >= 70 ? '#10b981' : tier.compliance_pct >= 40 ? '#f59e0b' : '#ef4444'
            return (
              <div key={i} style={{
                border: '1px solid #e2e8f0', borderRadius: 10, padding: 16,
                background: '#f8fafc'
              }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 8 }}>{tier.label}</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: compColor, marginBottom: 4 }}>{tier.compliance_pct}%</div>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>compliance</div>
                <div style={{ display: 'flex', gap: 12, fontSize: 12 }}>
                  <span style={{ color: STATUS_COLORS.present }}>Present: {tier.items_present}</span>
                  <span style={{ color: STATUS_COLORS.partial }}>Partial: {tier.items_partial}</span>
                  <span style={{ color: STATUS_COLORS.missing }}>Missing: {tier.items_missing}</span>
                </div>
                <div style={{
                  marginTop: 8, height: 6, borderRadius: 3, background: '#e2e8f0', overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%', width: `${tier.compliance_pct}%`, borderRadius: 3,
                    background: compColor, transition: 'width 0.3s'
                  }} />
                </div>
              </div>
            )
          })}
        </div>
      </Card>
    </div>
  )
}

function CategoryDetailTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {(data.categories || []).map((cat, ci) => (
        <Card key={ci} title={cat.category}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Item</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', width: 100 }}>Status</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Note</th>
              </tr>
            </thead>
            <tbody>
              {(cat.items || []).map((item, ii) => (
                <tr key={ii} style={{
                  background: item.status === 'present' ? '#f0fdf4' : item.status === 'missing' ? '#fef2f2' : undefined
                }}>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 500 }}>{item.name}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                    <StatusBadge status={item.status} />
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12, color: '#64748b' }}>
                    {item.note || '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}
    </div>
  )
}

function DataGapsTab({ data, breakdown }) {
  const topGaps = data.top_gaps || []

  // Group missing items by category
  const gapsByCategory = {}
  topGaps.forEach(g => {
    if (!gapsByCategory[g.category]) gapsByCategory[g.category] = []
    gapsByCategory[g.category].push(g)
  })

  // Also gather missing items from breakdown categories for completeness
  if (breakdown && breakdown.categories) {
    breakdown.categories.forEach(cat => {
      (cat.items || []).forEach(item => {
        if (item.status === 'missing') {
          if (!gapsByCategory[cat.category]) gapsByCategory[cat.category] = []
          const exists = gapsByCategory[cat.category].some(g => g.name === item.name)
          if (!exists) {
            gapsByCategory[cat.category].push({ name: item.name, category: cat.category, status: 'missing' })
          }
        }
      })
    })
  }

  const categoryKeys = Object.keys(gapsByCategory).sort()

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Missing Data Items (${data.missing_count} total)`}>
        <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
          Items that are not yet available and need to be acquired or generated.
        </p>
        {categoryKeys.map((cat, ci) => (
          <div key={ci} style={{ marginBottom: 20 }}>
            <div style={{
              fontSize: 13, fontWeight: 600, color: '#1e293b', marginBottom: 8,
              padding: '6px 12px', background: '#fef2f2', borderRadius: 6, borderLeft: '3px solid #ef4444'
            }}>
              {cat} ({gapsByCategory[cat].length} missing)
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', width: 40 }}>#</th>
                  <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Item Name</th>
                  <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', width: 100 }}>Priority</th>
                </tr>
              </thead>
              <tbody>
                {gapsByCategory[cat].map((g, gi) => {
                  // Items in top_gaps list are higher priority
                  const isTopGap = topGaps.some(tg => tg.name === g.name)
                  return (
                    <tr key={gi} style={{ background: '#fef2f2' }}>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #e2e8f0', color: '#94a3b8' }}>{gi + 1}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 500 }}>{g.name}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                          background: isTopGap ? '#fef2f2' : '#f8fafc',
                          color: isTopGap ? '#ef4444' : '#94a3b8'
                        }}>
                          {isTopGap ? 'High' : 'Normal'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ))}
      </Card>
    </div>
  )
}

function ControlGroupsTab({ data }) {
  const cg = data.control_groups || {}
  const minDataset = cg.minimum_dataset || []
  const idealDataset = cg.ideal_dataset || []
  const mostValuable = cg.most_valuable || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      {cg.note && (
        <Card title="Control Groups Overview" span={2}>
          <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{cg.note}</p>
        </Card>
      )}

      <Card title="Most Valuable Control Groups" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200, 1fr))', gap: 12 }}>
          {mostValuable.map((item, i) => (
            <div key={i} style={{
              border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, background: '#f0fdf4',
              fontSize: 13, color: '#1e293b', fontWeight: 500, textAlign: 'center'
            }}>
              {item}
            </div>
          ))}
        </div>
      </Card>

      <Card title="Minimum Dataset" span={1}>
        {minDataset.length > 0 ? (
          <ResponsiveContainer width="100%" height={Math.max(200, minDataset.length * 40)}>
            <BarChart data={minDataset} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" fontSize={12} label={{ value: 'n (subjects)', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis dataKey="cohort" type="category" fontSize={10} width={140} />
              <Tooltip formatter={(v) => [v, 'Subjects']} />
              <Bar dataKey="n" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Subjects" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ fontSize: 13, color: '#64748b' }}>No minimum dataset defined</p>
        )}
      </Card>

      <Card title="Ideal Dataset" span={1}>
        {idealDataset.length > 0 ? (
          <ResponsiveContainer width="100%" height={Math.max(200, idealDataset.length * 40)}>
            <BarChart data={idealDataset} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" fontSize={12} label={{ value: 'n (subjects)', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis dataKey="cohort" type="category" fontSize={10} width={140} />
              <Tooltip formatter={(v) => [v, 'Subjects']} />
              <Bar dataKey="n" fill="#10b981" radius={[0, 4, 4, 0]} name="Subjects" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ fontSize: 13, color: '#64748b' }}>No ideal dataset defined</p>
        )}
      </Card>

      {(data.artifact_template || []).length > 0 && (
        <Card title="Artifact Template" span={2}>
          {data.artifact_template.map((artCat, aci) => (
            <div key={aci} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b', marginBottom: 8 }}>{artCat.category}</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Type</th>
                    <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', width: 100 }}>Mandatory</th>
                  </tr>
                </thead>
                <tbody>
                  {(artCat.items || []).map((item, ii) => (
                    <tr key={ii}>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #e2e8f0' }}>{item.type}</td>
                      <td style={{ padding: '6px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                        color: item.mandatory ? '#ef4444' : '#64748b', fontWeight: item.mandatory ? 600 : 400 }}>
                        {item.mandatory ? 'Yes' : 'No'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Status Definitions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', width: 120 }}>Status</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.statuses || {}).map(([key, desc], i) => (
              <tr key={i}>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>
                  <StatusBadge status={key} />
                </td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Tier Descriptions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', width: 200 }}>Tier</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.tiers || {}).map(([key, desc], i) => (
              <tr key={i}>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#1e293b',
                  fontFamily: 'monospace', fontSize: 12 }}>{key}</td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Completeness Formula">
        <div style={{
          padding: 16, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0',
          fontFamily: 'monospace', fontSize: 13, color: '#334155', lineHeight: 1.6
        }}>
          {data.completeness_formula || '--'}
        </div>
      </Card>

      <Card title="Data Source">
        <p style={{ margin: 0, fontSize: 13, color: '#475569' }}>
          {data.data_source || 'Not specified'}
        </p>
      </Card>
    </div>
  )
}
