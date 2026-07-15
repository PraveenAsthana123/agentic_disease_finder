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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Flag Details' },
  { id: 'definitions', label: 'Definitions' },
]

export default function FeatureFlagsDashboard() {
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
      axios.get(`${API_URL}/api/feature-flags/overview`),
      axios.get(`${API_URL}/api/feature-flags/breakdown`),
      axios.get(`${API_URL}/api/feature-flags/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Feature Flags data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Feature Flags Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Flag lifecycle management — enabled/disabled status, rollout tiers, category breakdown, owner workload, staleness tracking
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const { kpis, category_distribution, rollout_tiers, owner_workload } = data

  const enabledData = [
    { name: 'Enabled', value: kpis.enabled },
    { name: 'Disabled', value: kpis.disabled },
  ]
  const enabledColors = ['#10b981', '#94a3b8']

  const catChart = (category_distribution || []).map(c => ({
    name: c.category,
    enabled: c.enabled,
    disabled: c.disabled,
  }))

  const tierChart = (rollout_tiers || []).map((t, i) => ({
    name: t.tier,
    count: t.count,
    fill: COLORS[i % COLORS.length],
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card>
        <KPI label="Total Flags" value={kpis.total_flags} />
      </Card>
      <Card>
        <KPI label="Enabled" value={kpis.enabled} sub={`${kpis.enabled_pct || Math.round(kpis.enabled / Math.max(kpis.total_flags, 1) * 100)}%`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Full Rollout" value={kpis.full_rollout} sub="at 100%" color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Stale Flags" value={kpis.stale_flags} sub=">90 days" color={kpis.stale_flags > 0 ? '#f59e0b' : '#10b981'} />
      </Card>

      <Card title="Enabled vs Disabled" span={2}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <ResponsiveContainer width="50%" height={200}>
            <PieChart>
              <Pie data={enabledData} dataKey="value" cx="50%" cy="50%" outerRadius={70} innerRadius={35} label={({ name, value }) => `${name}: ${value}`}>
                {enabledData.map((_, i) => <Cell key={i} fill={enabledColors[i]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div>
            <div style={{ fontSize: 13, marginBottom: 6 }}>
              <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#10b981', marginRight: 6 }} />
              Enabled: <strong>{kpis.enabled}</strong>
            </div>
            <div style={{ fontSize: 13, marginBottom: 6 }}>
              <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: '#94a3b8', marginRight: 6 }} />
              Disabled: <strong>{kpis.disabled}</strong>
            </div>
            <div style={{ fontSize: 13 }}>
              Avg rollout (enabled): <strong>{kpis.avg_rollout_pct}%</strong>
            </div>
          </div>
        </div>
      </Card>

      <Card title="Rollout Tier Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={tierChart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6">
              {tierChart.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Category Breakdown" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={catChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={100} />
            <Tooltip />
            <Legend />
            <Bar dataKey="enabled" stackId="a" fill="#10b981" name="Enabled" />
            <Bar dataKey="disabled" stackId="a" fill="#94a3b8" name="Disabled" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Owner Workload" span={2}>
        <div style={{ maxHeight: 220, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Owner</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Total</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Active</th>
              </tr>
            </thead>
            <tbody>
              {(owner_workload || []).map((o, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px' }}>{o.owner}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>{o.total_flags}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px', color: '#10b981', fontWeight: 600 }}>{o.active}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  const { all_flags, stale_flags, disabled_flags, rollout_candidates, by_category } = data

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="All Feature Flags" span={1}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Name</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Status</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Rollout</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Category</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Owner</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Age (d)</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {(all_flags || []).map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '5px 8px', fontFamily: 'monospace', fontSize: 11 }}>{f.flag_id}</td>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{f.name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <span style={{
                      display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                      background: f.enabled ? '#dcfce7' : '#f1f5f9',
                      color: f.enabled ? '#16a34a' : '#64748b',
                    }}>{f.enabled ? 'ON' : 'OFF'}</span>
                  </td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'center' }}>
                      <div style={{ width: 40, height: 6, borderRadius: 3, background: '#e2e8f0', overflow: 'hidden' }}>
                        <div style={{ width: `${f.rollout_percentage}%`, height: '100%', background: f.rollout_percentage === 100 ? '#10b981' : '#3b82f6', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11 }}>{f.rollout_percentage}%</span>
                    </div>
                  </td>
                  <td style={{ padding: '5px 8px' }}>
                    <span style={{ padding: '1px 6px', borderRadius: 6, fontSize: 10, background: '#eff6ff', color: '#3b82f6' }}>{f.category}</span>
                  </td>
                  <td style={{ padding: '5px 8px', fontSize: 11 }}>{f.owner}</td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: f.days_since_update > 90 ? '#ef4444' : '#64748b' }}>{f.days_since_update ?? '-'}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {rollout_candidates && rollout_candidates.length > 0 && (
        <Card title={`Rollout Candidates (${rollout_candidates.length} enabled, not at 100%)`}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {rollout_candidates.map((f, i) => (
              <div key={i} style={{ padding: '8px 14px', background: '#eff6ff', borderRadius: 8, fontSize: 12 }}>
                <strong>{f.name}</strong>
                <span style={{ marginLeft: 8, color: '#3b82f6' }}>{f.rollout_percentage}%</span>
                <span style={{ marginLeft: 6, color: '#64748b' }}>{f.owner}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {stale_flags && stale_flags.length > 0 && (
        <Card title={`Stale Flags (${stale_flags.length} not updated in 90+ days)`}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Name</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Status</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Days Stale</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Owner</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Last Updated</th>
              </tr>
            </thead>
            <tbody>
              {stale_flags.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{f.name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                      background: f.enabled ? '#dcfce7' : '#f1f5f9',
                      color: f.enabled ? '#16a34a' : '#64748b',
                    }}>{f.enabled ? 'ON' : 'OFF'}</span>
                  </td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', color: '#ef4444', fontWeight: 600 }}>{f.days_since_update}</td>
                  <td style={{ padding: '5px 8px' }}>{f.owner}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{f.updated_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {disabled_flags && disabled_flags.length > 0 && (
        <Card title={`Disabled Flags — Cleanup Candidates (${disabled_flags.length})`}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Name</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Category</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Owner</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Age (d)</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {disabled_flags.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{f.name}</td>
                  <td style={{ padding: '5px 8px' }}>{f.category}</td>
                  <td style={{ padding: '5px 8px' }}>{f.owner}</td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11 }}>{f.age_days ?? '-'}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{f.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {by_category && Object.keys(by_category).length > 0 && (
        <Card title="Flags by Category">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
            {Object.entries(by_category).map(([cat, flags]) => (
              <div key={cat} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>{cat} ({flags.length})</h4>
                {flags.map((f, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0', fontSize: 12 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: '50%',
                      background: f.enabled ? '#10b981' : '#94a3b8',
                    }} />
                    <span style={{ flex: 1 }}>{f.name}</span>
                    <span style={{ color: '#3b82f6', fontSize: 11 }}>{f.rollout_percentage}%</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ data }) {
  const { statuses, rollout_tiers, staleness_thresholds, categories, best_practices } = data || {}

  const sections = [
    { title: 'Flag Statuses', items: statuses ? Object.entries(statuses).map(([k, v]) => ({ term: k, definition: v })) : [] },
    { title: 'Rollout Tiers', items: rollout_tiers ? Object.entries(rollout_tiers).map(([k, v]) => ({ term: k, definition: v })) : [] },
    { title: 'Staleness Thresholds', items: staleness_thresholds ? Object.entries(staleness_thresholds).map(([k, v]) => ({ term: k, definition: v })) : [] },
    { title: 'Categories', items: categories ? Object.entries(categories).map(([k, v]) => ({ term: k, definition: v })) : [] },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map((s, si) => (
        <Card key={si} title={s.title}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <tbody>
              {s.items.map((item, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155' }}>{item.term}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{item.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}

      {best_practices && best_practices.length > 0 && (
        <Card title="Best Practices" span={2}>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#334155' }}>
            {best_practices.map((bp, i) => (
              <li key={i} style={{ marginBottom: 4 }}>{bp}</li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}
